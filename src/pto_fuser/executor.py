"""Staged executor — the M1 baseline backend.

Runs a `Program` node-by-node over a name->tensor environment. Each `EinsumNode`
dispatches to its own substrate `.so` (persistent-workspace setup/exec/teardown,
handled inside `pto_einsum`); stages share GM tensors through the environment.
This is the correctness reference every other backend (graph-replay, fused-node)
is gated against (docs/FUSER_DESIGN.md §5–6).

The executor honors only the **default** einsum lowering (NN, no fused store, no
folded glue). Planner annotations on `EinsumNode` are intentionally ignored here;
they become active backends in M2+.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Callable, Dict, Optional

import torch

from .fused import FusedKernelRegistry, shared_fused_registry
from .ir import (EinsumNode, FusedNode, OpaqueNode, Program, TensorOp,
                 VecGlueNode, node_outputs)
from .registry import OpaqueRegistry, default_registry


@contextlib.contextmanager
def substrate_modes(read_mode: str = "auto", fuse_out: bool = True):
    """Drive the substrate's documented lowering toggles for one einsum build.

    The substrate auto-selects the read mode and the operand-swap-to-fused-store
    from the equation+layout; these env knobs only let the fuser *forbid* an
    optimization (to fall back to the always-valid baseline). `read_mode="NN"`
    forces Phase-A (no direct read); `fuse_out=False` forbids the operand swap.
    The defaults (`"auto"`, `True`) set nothing — the substrate decides.
    """
    overrides = {}
    if read_mode == "NN":
        overrides["EINSUM_DISABLE_NT"] = "1"
    if not fuse_out:
        overrides["EINSUM_DISABLE_OPERAND_SWAP"] = "1"
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _load_substrate_einsum() -> Callable:
    """Import `einsum` from the pinned pto-einsum substrate (sibling repo by
    default; override with PTO_EINSUM)."""
    einsum_root = os.environ.get(
        "PTO_EINSUM",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "pto-einsum"))
    src = os.path.join(einsum_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pto_einsum import einsum  # noqa: import-after-path
    return einsum


def _cast(t: torch.Tensor, dtype) -> torch.Tensor:
    return t if dtype is None or t.dtype == dtype else t.to(dtype)


class StagedExecutor:
    def __init__(self, registry: Optional[OpaqueRegistry] = None,
                 einsum_fn: Optional[Callable] = None,
                 fused: Optional[FusedKernelRegistry] = None) -> None:
        self.registry = registry or default_registry()
        self.fused = fused or shared_fused_registry()   # lever-5/6 fused kernels (M4)
        self._einsum = einsum_fn          # lazily resolved so import works off-NPU

    @property
    def einsum(self) -> Callable:
        if self._einsum is None:
            self._einsum = _load_substrate_einsum()
        return self._einsum

    def run(self, program: Program, bindings: Dict[str, torch.Tensor],
            return_env: bool = False) -> Dict[str, torch.Tensor]:
        missing = [n for n in program.inputs if n not in bindings]
        if missing:
            raise ValueError(f"missing bindings for inputs: {missing}")
        env: Dict[str, torch.Tensor] = dict(bindings)
        for node in program.nodes:
            result = self._exec(node, env)
            if isinstance(result, dict):                     # FusedNode: many outputs
                env.update(result)
            else:
                env[node.output] = result
        if return_env:                                       # planner needs the
            return env                                       # staged intermediates
        return {name: env[name] for name in program.outputs}

    # -- per-node dispatch -------------------------------------------------- #
    def _exec(self, node, env: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(node, EinsumNode):
            a, b = (env[n] for n in node.inputs)
            with substrate_modes(node.read_mode, node.fuse_out):
                out = self.einsum(node.equation, a, b)       # substrate -> fp32
            return _cast(out, node.out_dtype)
        if isinstance(node, OpaqueNode):
            ins = [env[n] for n in node.inputs]
            return self.registry.run(node.kernel, ins, node.params)
        if isinstance(node, FusedNode):
            ins = [env[n] for n in node.inputs]
            outs = self.fused.run(node.kernel, ins, node.params)
            return {name: t for name, t in zip(node.outputs, outs)}
        if isinstance(node, VecGlueNode):
            return self._exec_glue(node, env)
        if isinstance(node, TensorOp):
            return self._exec_tensorop(node, env)
        raise TypeError(f"unknown node type: {type(node).__name__}")

    def _exec_glue(self, node: VecGlueNode, env) -> torch.Tensor:
        op, p = node.op, node.params
        ins = [env[n].float() for n in node.inputs]         # accumulate in fp32
        if op == "tril":
            out = torch.tril(ins[0], diagonal=p.get("diagonal", 0))
        elif op == "add":
            out = ins[0] + ins[1]
        elif op == "sub":
            out = ins[0] - ins[1]
        elif op == "mul":
            out = ins[0] * ins[1]                            # broadcasting allowed
        elif op == "scale":
            out = ins[0] * p["scalar"]
        else:
            raise ValueError(f"unknown glue op {op!r}")
        return _cast(out, node.out_dtype)

    def _exec_tensorop(self, node: TensorOp, env) -> torch.Tensor:
        op, p = node.op, node.params
        if op == "zeros":
            ref = env[node.inputs[0]] if node.inputs else None
            device = p.get("device") or (ref.device if ref is not None else None)
            return torch.zeros(p["shape"], dtype=p["dtype"], device=device)
        x = env[node.inputs[0]]
        if op == "reshape":
            return x.reshape(p["shape"])
        if op == "contiguous":
            return x.contiguous()
        if op == "transpose":
            return x.transpose(*p["dims"])
        if op == "permute":
            return x.permute(*p["dims"]).contiguous()
        if op == "cast":
            return x.to(p["dtype"])
        if op == "slice":
            return x.select(p["axis"], p["index"])
        if op == "stack":
            return torch.stack([env[n] for n in node.inputs], dim=p["dim"])
        raise ValueError(f"unknown tensor op {op!r}")
