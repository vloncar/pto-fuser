"""pto-fuser IR — the three compute node types + host plumbing.

A program is an ordered list of steps over named tensors (an environment that
maps name -> torch.Tensor). The design (docs/DESIGN.md §3) fixes **three
compute node types**, proved sufficient by the chunk-attention taxonomy:

  * ``EinsumNode``   — one library contraction (the core unit).
  * ``OpaqueNode``   — a foreign hand-optimized kernel the matmul-core can't express.
  * ``VecGlueNode``  — a standalone Vector op (mask / scale / elementwise residual).

`TensorOp` is **not** a fourth compute type: it is host tensor plumbing — views,
reshapes, casts, slices, stacks, allocations — that a *staged* executor needs to
wire intermediates between kernels. It carries no device kernel. Under graph capture these collapse into buffer-binding metadata, not ops; they live
in the IR only because the staged executor is host-driven.

`read_mode` / `fuse_out` / `epilogue` / `prologue` on ``EinsumNode`` are **planner
outputs**, not user input. The default lowering is always a correct execution; the
The planner sets `read_mode` / `fuse_out` per node by measuring the library's
direct-read / fused-store lowering against the always-valid Phase-A NN baseline and
keeping it only when gated-green and faster (see ``planner.py``). The executor
honors these annotations via the library's documented mode toggles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class TensorRef:
    """A named handle into the execution environment.

    `shape`/`dtype` are optional and advisory (documentation + future validation);
    the executor binds names to concrete tensors at run time.
    """
    name: str
    shape: Optional[tuple] = None
    dtype: Optional[Any] = None

    def __str__(self) -> str:  # so a ref can stand in for its name in messages
        return self.name


class Node:
    """Base for every program step (compute node or host TensorOp)."""
    inputs: List[str]
    output: str


@dataclass
class EinsumNode(Node):
    """One library contraction. `inputs` are exactly two operand names."""
    equation: str
    inputs: List[str]
    output: str
    out_dtype: Optional[Any] = None     # cast the (fp32) library result; None = native
    # --- planner annotations (set + gated by the Planner; see planner.py) ---
    # The library auto-selects the read mode (NT/NN-strided/TN, the read/store modes–2.13) and the
    # operand-swap-to-fused-store (the read/store modes) from the equation+layout; it exposes the
    # documented toggles EINSUM_DISABLE_NT / EINSUM_DISABLE_OPERAND_SWAP. These two
    # fields *select among* those (see DESIGN.md), they do not re-implement them:
    #   read_mode = "auto" -> let the library pick the direct-read mode (default);
    #               "NN"   -> force the always-valid Phase-A NN lowering.
    #   fuse_out  = True   -> permit the operand-swap that exposes the fused store
    #               (default); False -> forbid the swap. The plain fused store still
    #               auto-fires when free1 is already innermost regardless of this.
    read_mode: str = "auto"             # auto | NN   (NT/NN_strided/TN chosen by library)
    fuse_out: bool = True               # the read/store modes / operand-swap permitted
    epilogue: Optional[list] = None     # glue absorbed into the store (planner detects; fused-node backend emits)
    prologue: Optional[list] = None     # per-operand scaling folded into the load


@dataclass
class OpaqueNode(Node):
    """A foreign hand-optimized kernel, hosted by key from the opaque registry.

    The registry entry owns the pinned contract — required input dtype/layout,
    adapters (e.g. transpose-to-upper for tri_inv), and the dtype cast. Dtype is
    part of the contract: the single hardest e2e bug was an fp16-only kernel handed
    fp32 bytes -> deterministic NaN, so the lowering inserts the cast explicitly.
    """
    kernel: str
    inputs: List[str]
    output: str
    params: dict = field(default_factory=dict)


@dataclass
class VecGlueNode(Node):
    """A standalone Vector op not (yet) absorbed into an adjacent contraction.

    The staged executor lowers these via torch (host-side); the glue-absorption pass is
    what folds bandwidth-bound glue into an `EinsumNode` epilogue/prologue. `op`
    is one of: tril | add | sub | mul | scale.
    """
    op: str
    inputs: List[str]
    output: str
    params: dict = field(default_factory=dict)
    out_dtype: Optional[Any] = None     # glue computes in fp32, casts to this


@dataclass
class FusedNode(Node):
    """A sub-chain collapsed into ONE hand-fused kernel, hosted by key from the
    fused-kernel registry (``fused.py``). This is the fused-node backend (see DESIGN.md):
    a resident-state recurrence (``chunk_h_scan``) or a matmul-core + on-chip
    epilogue (``kkt_gated``) that subsumes several staged stages into a single
    dispatch, keeping intermediates (the carried state, the gated qk) on-chip
    instead of round-tripping HBM.

    Unlike the staged lowering it replaces, a fused kernel may produce **several**
    outputs (the scan emits both per-chunk readouts and the final state), so this is
    the one node type with an ``outputs`` *list*. It is kept only when the verifier
    (``fusion.py`` / ``compile.py``) gates it bit-faithful + deterministic vs the
    canonical lowering AND measures it faster (the determinism gate is mandatory on any
    fused lowering).

    The hosted kernels live in ``kernels/`` (``kkt_fused``, ``chunk_h_scan``), built as
    their own ``.so`` and sharing GM with the surrounding stages — the form that
    *works today*. The
    further step of inlining an opaque AICORE device-fn into one ``.so`` with the
    library matmul core (the tri_inv case) stays unproven research; it is not used
    here. ``inputs`` carry the logical operand names; the registry lowering owns the
    layout/dtype adapters, exactly as the opaque registry does for tri_inv.
    """
    kernel: str
    inputs: List[str]
    outputs: List[str]
    params: dict = field(default_factory=dict)
    subsumes: List[str] = field(default_factory=list)   # staged outputs this replaces (doc/decision)

    @property
    def output(self) -> str:                            # primary, for messages
        return self.outputs[0]


@dataclass
class TensorOp(Node):
    """Host tensor plumbing (no kernel): reshape | contiguous | transpose | cast |
    slice | stack | zeros. See module docstring — not a compute node type."""
    op: str
    inputs: List[str]
    output: str
    params: dict = field(default_factory=dict)


@dataclass
class Program:
    """An ordered list of steps plus the declared input/output names.

    `inputs` are bound at run time; `outputs` are returned by the executor.
    The order is the execution order (the unrolled static chain — exactly the
    shape graph capture wants).
    """
    nodes: List[Node]
    inputs: List[str]
    outputs: List[str]

    def __post_init__(self) -> None:
        # cheap structural check: every consumed name is either an input or
        # produced by an earlier step (catches build-order bugs early).
        produced = set(self.inputs)
        for node in self.nodes:
            for name in node.inputs:
                if name not in produced:
                    raise ValueError(
                        f"step producing {node.output!r} reads {name!r} before it is "
                        f"produced or bound as an input")
            produced.update(node_outputs(node))   # FusedNode produces several
        missing = [o for o in self.outputs if o not in produced]
        if missing:
            raise ValueError(f"declared outputs never produced: {missing}")


def node_outputs(node: Node) -> List[str]:
    """The names a step binds into the environment. A `FusedNode` produces an
    `outputs` list; every other node produces its single `output`."""
    outs = getattr(node, "outputs", None)
    return list(outs) if outs is not None else [node.output]
