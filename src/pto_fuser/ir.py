"""pto-fuser IR — the three compute node types + host plumbing.

A program is an ordered list of steps over named tensors (an environment that
maps name -> torch.Tensor). The design (docs/FUSER_DESIGN.md §3) fixes **three
compute node types**, proved sufficient by the chunk-attention taxonomy:

  * ``EinsumNode``   — one substrate contraction (the core unit).
  * ``OpaqueNode``   — a foreign hand-optimized kernel the matmul-core can't express.
  * ``VecGlueNode``  — a standalone Vector op (mask / scale / elementwise residual).

`TensorOp` is **not** a fourth compute type: it is host tensor plumbing — views,
reshapes, casts, slices, stacks, allocations — that a *staged* executor needs to
wire intermediates between kernels. It carries no device kernel. Under graph
capture (M3) these collapse into buffer-binding metadata, not ops; they live in
the IR now only because M1 is a host-driven staged executor.

`read_mode` / `fuse_out` / `epilogue` / `prologue` on ``EinsumNode`` are **planner
outputs**, not user input. The default lowering (NN, no fuse, no folded glue) is
always a correct execution; M1 honors only the default and the planner (M2+) fills
these in as gated annotations.
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
    """One substrate contraction. `inputs` are exactly two operand names."""
    equation: str
    inputs: List[str]
    output: str
    out_dtype: Optional[Any] = None     # cast the (fp32) substrate result; None = native
    # --- planner annotations (M1: defaults only; M2 populates + gates) ---
    read_mode: str = "NN"               # NN | NT | NN_strided | TN  (§2.11–2.13)
    fuse_out: bool = False              # §2.9 fused permuted store
    epilogue: Optional[list] = None     # glue folded into the store
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

    M1 lowers these via torch (host-side); M2's glue-absorption pass (lever 4) is
    what folds bandwidth-bound glue into an `EinsumNode` epilogue/prologue. `op`
    is one of: tril | add | sub | mul | scale.
    """
    op: str
    inputs: List[str]
    output: str
    params: dict = field(default_factory=dict)
    out_dtype: Optional[Any] = None     # glue computes in fp32, casts to this


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
    shape graph capture wants in M3).
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
            produced.add(node.output)
        missing = [o for o in self.outputs if o not in produced]
        if missing:
            raise ValueError(f"declared outputs never produced: {missing}")
