"""Transforms — the *what* of optimization, separated from the *whether*.

A `Transform` is a pure ``Program -> Program`` rewrite: it matches a structural
pattern in the IR and rewrites it, and it does **nothing else**. It does not
measure, it does not decide, it does not touch the device. Whether a transform is
worth applying is the job of the cost model (`cost.py`) and the policy
(`policy.py`); whether a proposed rewrite is *correct and faster* is the job of the
verifier (`compile.py`, reusing the `fusion`/`gate` machinery). This three-way split
— transform / heuristic / verification — is the point of this module: the old
per-lever code baked all three into one gated build-flag branch, which is why adding
a lever meant threading a new flag through the builder, the executor, and every
benchmark.

The design borrows the `cce-mlir` pass shape (a named pass with a summary, a match
predicate, and typed *options* — cf. `PtoMixSplit`'s opt-in marker and
`LowerPTOToCCE`'s options): every `Transform` carries a kebab-case ``name``, a
one-line ``summary``, an optional set of construction options (the forward's dims),
a ``match`` predicate returning the number of rewritable sites, and an ``apply`` that
returns a `TransformResult`. Transforms compose (a pipeline is an ordered list) and
each is independently verifiable against the program it was handed.

Two universal transforms live here (they match any `EinsumNode`): the read-mode and
fused-store *selection* levers, realized as annotation rewrites over the always-valid
NN baseline that `canonicalize` establishes. The structural fusion transforms
(resident-state scan, glue absorption) are forward-shaped and live in
`transforms/`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from .ir import EinsumNode, Node, Program, node_outputs


# --------------------------------------------------------------------------- #
#  result + base
# --------------------------------------------------------------------------- #
@dataclass
class TransformResult:
    """The outcome of applying one transform: the rewritten program, whether it
    changed anything, how many sites it rewrote, and a human-readable note."""
    program: Program
    changed: bool
    sites: int
    note: str = ""


class Transform:
    """A pure IR->IR rewrite. Subclasses set ``name``/``summary`` and implement
    ``match`` (how many sites are rewritable, 0 = not applicable) and ``apply``
    (the rewrite). ``apply`` must be a pure function of the program (+ any options
    fixed at construction) — no measurement, no device, no global state."""

    name: str = "transform"
    summary: str = ""

    def match(self, program: Program) -> int:      # pragma: no cover - abstract
        raise NotImplementedError

    def apply(self, program: Program) -> TransformResult:   # pragma: no cover
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
#  IR helpers (shared by the structural transforms)
# --------------------------------------------------------------------------- #
def producer_index(program: Program) -> Dict[str, int]:
    """name -> index of the node that produces it (last writer wins)."""
    out: Dict[str, int] = {}
    for i, node in enumerate(program.nodes):
        for nm in node_outputs(node):
            out[nm] = i
    return out


def consumer_map(program: Program) -> Dict[str, List[int]]:
    """name -> indices of nodes that read it."""
    out: Dict[str, List[int]] = {}
    for i, node in enumerate(program.nodes):
        for nm in node.inputs:
            out.setdefault(nm, []).append(i)
    return out


def splice(program: Program, drop: set, insert_at: int,
           new_nodes: List[Node]) -> Program:
    """Return a program with the nodes at indices ``drop`` removed and
    ``new_nodes`` inserted so that the first survivor at/after ``insert_at`` keeps
    its relative order. Inputs/outputs are preserved. The caller guarantees the
    result is still a valid use-before-def ordering (checked by
    ``Program.__post_init__``)."""
    nodes: List[Node] = []
    inserted = False
    for i, node in enumerate(program.nodes):
        if i == insert_at:
            nodes.extend(new_nodes)
            inserted = True
        if i not in drop:
            nodes.append(node)
    if not inserted:
        nodes.extend(new_nodes)
    return Program(nodes=nodes, inputs=list(program.inputs),
                   outputs=list(program.outputs))


# --------------------------------------------------------------------------- #
#  canonical form
# --------------------------------------------------------------------------- #
def canonicalize(program: Program) -> Program:
    """The always-valid baseline every transform starts from: every `EinsumNode`
    forced to the Phase-A **NN** read with **no** fused store. This is the lowering
    the `StagedExecutor` honors unconditionally and the correctness reference the
    verifier gates against. Read-mode / fused-store selection is then reintroduced
    as the `EnableDirectReads` / `EnableFusedStore` transforms — so those levers are
    *optimizations proven against this baseline*, not the default."""
    nodes: List[Node] = []
    for node in program.nodes:
        if isinstance(node, EinsumNode):
            node = replace(node, read_mode="NN", fuse_out=False)
        nodes.append(node)
    return Program(nodes=nodes, inputs=list(program.inputs),
                   outputs=list(program.outputs))


# --------------------------------------------------------------------------- #
#  universal selection transforms (read-mode / fused-store)
# --------------------------------------------------------------------------- #
class EnableDirectReads(Transform):
    """Let the library pick its direct-read mode (NT / NN-strided / TN) instead of
    the Phase-A NN baseline, on every `EinsumNode` still reading NN. Pure
    annotation: sets ``read_mode="auto"``. The library auto-selects *which* mode
    fires from the operand layout; whether it is faster than NN is the verifier's
    call (it is layout-dependent — huge on the head-strided GDN family, ~1.0x on the
    flat DeltaNet family)."""

    name = "enable-direct-reads"
    summary = "EinsumNode read_mode NN -> auto (library direct read)"

    def match(self, program: Program) -> int:
        return sum(isinstance(n, EinsumNode) and n.read_mode == "NN"
                   for n in program.nodes)

    def apply(self, program: Program) -> TransformResult:
        nodes, sites = [], 0
        for node in program.nodes:
            if isinstance(node, EinsumNode) and node.read_mode == "NN":
                node = replace(node, read_mode="auto")
                sites += 1
            nodes.append(node)
        prog = Program(nodes=nodes, inputs=list(program.inputs),
                       outputs=list(program.outputs))
        return TransformResult(prog, sites > 0, sites,
                               f"{sites} einsum(s) -> direct read")


class EnableFusedStore(Transform):
    """Permit the operand-swap that exposes the fused permuted store on every
    `EinsumNode` with it disabled. Pure annotation: sets ``fuse_out=True``. The plain
    fused store still auto-fires when free1 is already innermost; this only *permits*
    the swap that can make it innermost."""

    name = "enable-fused-store"
    summary = "EinsumNode fuse_out False -> True (operand-swap fused store)"

    def match(self, program: Program) -> int:
        return sum(isinstance(n, EinsumNode) and not n.fuse_out
                   for n in program.nodes)

    def apply(self, program: Program) -> TransformResult:
        nodes, sites = [], 0
        for node in program.nodes:
            if isinstance(node, EinsumNode) and not node.fuse_out:
                node = replace(node, fuse_out=True)
                sites += 1
            nodes.append(node)
        prog = Program(nodes=nodes, inputs=list(program.inputs),
                       outputs=list(program.outputs))
        return TransformResult(prog, sites > 0, sites,
                               f"{sites} einsum(s) -> fused store permitted")
