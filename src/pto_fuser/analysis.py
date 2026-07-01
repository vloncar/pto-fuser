"""Fusion-region identification — the first step toward megakernel generation.

This is an **analysis**, not a rewrite: it partitions a canonical `Program` into the
maximal scopes that *could* become a single kernel, scores each by the HBM traffic that
fusing it would keep on-chip (the L2-residency opportunity), and classifies it — and it
emits nothing. No codegen, no device: the sizing is done by propagating shapes over
``torch`` **meta** tensors (a pure shape/dtype walk, never a real kernel), so the whole
pass runs off-NPU.

Why this is the honest first megakernel step: the hardest judgement in generating a
fused kernel is *what to fuse*. Doing it as a separable, reviewable analysis — before a
single kernel is emitted — makes that judgement explicit and lets us prove a safety
property up front: **a fusion never spans an opaque-kernel boundary** (the triangular
inverse), and every structural transform we already apply stays inside one identified
region. A later template-emission step (§8 of ``docs/DESIGN.md``) picks kernel
boundaries *within* a region; this pass finds the regions and quantifies the prize.

The region representation is deliberately compiler-shaped (a named analysis over the IR
producing regions with explicit boundary in/out and internal sets) so it maps onto a
`cce-mlir` fusion/outlining analysis when this stack is reimplemented there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Optional

import torch

from .ir import (EinsumNode, FusedNode, OpaqueNode, Program, TensorOp,
                 VecGlueNode, node_outputs)
from .transform import producer_index


# --------------------------------------------------------------------------- #
#  region model
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class FusionRegion:
    """A maximal fusible scope — a set of nodes that a single kernel could subsume.

    ``inputs`` are read from outside (unavoidable loads), ``outputs`` are consumed
    outside or are program outputs (unavoidable stores), and ``internal`` tensors are
    produced *and* consumed within the region — the ones a fused kernel would keep
    on-chip instead of round-tripping HBM. ``internal_bytes`` is therefore the
    L2-residency opportunity (an upper bound on the HBM traffic fusion removes)."""
    id: int
    kind: str                              # recurrence | contraction-epilogue | elementwise | plumbing
    nodes: List[int]                       # indices into program.nodes
    node_names: List[str]                  # their output names (readable)
    inputs: List[str]
    outputs: List[str]
    internal: List[str]
    internal_bytes: int
    boundary_in_bytes: int
    boundary_out_bytes: int
    device_dispatches: int                 # einsum/glue/fused nodes (tensorops fold under capture)
    n_einsum: int
    n_glue: int
    has_recurrence: bool

    @property
    def score(self) -> int:
        """Ranking proxy: the on-chip bytes fusion would save."""
        return self.internal_bytes

    def __str__(self) -> str:
        mb = self.internal_bytes / 1e6
        io = (self.boundary_in_bytes + self.boundary_out_bytes) / 1e6
        return (f"region {self.id} [{self.kind:<20}] "
                f"{len(self.nodes):3d} nodes / {self.device_dispatches:2d} dispatches "
                f"({self.n_einsum} einsum, {self.n_glue} glue)  "
                f"internal {mb:7.2f} MB on-chip  | boundary I/O {io:7.2f} MB")


@dataclass
class FusionAnalysis:
    regions: List[FusionRegion]
    opaque: List[str] = field(default_factory=list)      # opaque-kernel boundary node names

    def ranked(self) -> List[FusionRegion]:
        return sorted(self.regions, key=lambda r: r.score, reverse=True)

    @property
    def total_internal_bytes(self) -> int:
        return sum(r.internal_bytes for r in self.regions)

    def region_of(self, node_name: str) -> Optional[FusionRegion]:
        for r in self.regions:
            if node_name in r.node_names:
                return r
        return None

    def __str__(self) -> str:
        head = (f"fusion-region analysis — {len(self.regions)} fusible region(s), "
                f"{len(self.opaque)} opaque boundary node(s); "
                f"{self.total_internal_bytes / 1e6:.2f} MB fusible on-chip total")
        lines = [head] + [str(r) for r in self.ranked()]
        if self.opaque:
            lines.append(f"opaque boundaries (hard fusion cuts): {', '.join(self.opaque)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  the analysis
# --------------------------------------------------------------------------- #
def identify_fusion_regions(program: Program, bindings: Dict[str, torch.Tensor]
                            ) -> FusionAnalysis:
    """Partition ``program`` into fusible regions, cut at opaque-kernel boundaries.

    ``bindings`` supplies the program inputs; only their ``.shape``/``.dtype`` are read
    (host-side, no device sync) to seed a meta-tensor shape walk that sizes every
    intermediate. Returns a `FusionAnalysis`."""
    shapes = _infer_shapes(program, bindings)
    prod = producer_index(program)
    consumers = _consumer_map(program)
    n = len(program.nodes)
    opaque_idx = {i for i, nd in enumerate(program.nodes) if isinstance(nd, OpaqueNode)}

    # union adjacent non-opaque nodes; never fuse across an opaque boundary.
    uf = _UnionFind(n)
    for j, nd in enumerate(program.nodes):
        if j in opaque_idx:
            continue
        for nm in nd.inputs:
            i = prod.get(nm)
            if i is None or i in opaque_idx:
                continue
            uf.union(i, j)

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        if i in opaque_idx:
            continue
        comps.setdefault(uf.find(i), []).append(i)

    regions: List[FusionRegion] = []
    for rid, idxs in enumerate(sorted(comps.values(), key=min)):
        regions.append(_assemble_region(rid, idxs, program, consumers, shapes))
    opaque_names = [program.nodes[i].output for i in sorted(opaque_idx)]
    return FusionAnalysis(regions, opaque_names)


def _assemble_region(rid, idxs, program, consumers, shapes) -> FusionRegion:
    members = set(idxs)
    produced, consumed = set(), set()
    n_einsum = n_glue = dispatches = 0
    has_recurrence = False
    for i in idxs:
        nd = program.nodes[i]
        produced.update(node_outputs(nd))
        consumed.update(nd.inputs)
        if isinstance(nd, EinsumNode):
            n_einsum += 1; dispatches += 1
        elif isinstance(nd, VecGlueNode):
            n_glue += 1; dispatches += 1
        elif isinstance(nd, FusedNode):
            dispatches += 1
        elif isinstance(nd, TensorOp) and nd.op == "zeros":
            has_recurrence = True          # a zero-seeded carried state = a scan

    inputs = sorted(consumed - produced)                       # read from outside
    outputs, internal = [], []
    for nm in sorted(produced):
        consumed_outside = any(c not in members for c in consumers.get(nm, []))
        is_program_out = nm in program.outputs
        if consumed_outside or is_program_out:
            outputs.append(nm)
        elif nm in consumed:                                   # produced+consumed inside
            internal.append(nm)
        else:
            outputs.append(nm)                                 # dead-but-produced: treat as boundary

    return FusionRegion(
        id=rid,
        kind=_classify(n_einsum, n_glue, has_recurrence),
        nodes=sorted(idxs),
        node_names=sorted(produced),
        inputs=inputs, outputs=outputs, internal=internal,
        internal_bytes=sum(_bytes(shapes.get(nm)) for nm in internal),
        boundary_in_bytes=sum(_bytes(shapes.get(nm)) for nm in inputs),
        boundary_out_bytes=sum(_bytes(shapes.get(nm)) for nm in outputs),
        device_dispatches=dispatches, n_einsum=n_einsum, n_glue=n_glue,
        has_recurrence=has_recurrence)


def _classify(n_einsum, n_glue, has_recurrence) -> str:
    if has_recurrence:
        return "recurrence"
    if n_einsum:
        return "contraction-epilogue"
    if n_glue:
        return "elementwise"
    return "plumbing"


# --------------------------------------------------------------------------- #
#  meta-tensor shape/dtype inference (pure; no device, no real kernel)
# --------------------------------------------------------------------------- #
def _infer_shapes(program: Program, bindings: Dict[str, torch.Tensor]
                  ) -> Dict[str, torch.Tensor]:
    """name -> a meta tensor carrying its shape+dtype. Seeds from the bindings'
    shapes/dtypes (read host-side) and propagates over meta ops."""
    env: Dict[str, torch.Tensor] = {}
    for name, t in bindings.items():
        env[name] = torch.empty(tuple(t.shape), dtype=t.dtype, device="meta")

    def meta(shape, dtype):
        return torch.empty(tuple(shape), dtype=dtype, device="meta")

    for nd in program.nodes:
        if isinstance(nd, EinsumNode):
            a, b = (env[x] for x in nd.inputs)
            out = torch.einsum(nd.equation, a, b)
            env[nd.output] = meta(out.shape, nd.out_dtype or a.dtype)
        elif isinstance(nd, OpaqueNode):
            x = env[nd.inputs[0]]
            env[nd.output] = meta(x.shape, torch.float32)      # tri_inv contract: fp32 out
        elif isinstance(nd, VecGlueNode):
            ins = [env[x] for x in nd.inputs]
            if nd.op == "tril":
                shape = ins[0].shape
            elif nd.op == "scale":
                shape = ins[0].shape
            else:                                              # mul / add / sub broadcast
                shape = torch.broadcast_shapes(*[t.shape for t in ins])
            env[nd.output] = meta(shape, nd.out_dtype or ins[0].dtype)
        elif isinstance(nd, TensorOp):
            env[nd.output] = _meta_tensorop(nd, env, meta)
        elif isinstance(nd, FusedNode):
            # canonical programs carry none; size its outputs from any bound shapes if present.
            for o in nd.outputs:
                env.setdefault(o, meta((), torch.float16))
    return env


def _meta_tensorop(nd: TensorOp, env, meta) -> torch.Tensor:
    op, p = nd.op, nd.params
    if op == "zeros":
        return meta(p["shape"], p["dtype"])
    x = env[nd.inputs[0]]
    if op == "reshape":
        return x.reshape(tuple(p["shape"]))
    if op in ("contiguous",):
        return x
    if op == "cast":
        return meta(x.shape, p["dtype"])
    if op == "transpose":
        return x.transpose(*p["dims"])
    if op == "permute":
        return x.permute(*p["dims"])
    if op == "slice":
        return x.select(p["axis"], p["index"])
    if op == "stack":
        return torch.stack([env[i] for i in nd.inputs], dim=p["dim"])
    raise ValueError(f"meta shape rule missing for tensor op {op!r}")


def _bytes(t: Optional[torch.Tensor]) -> int:
    if t is None:
        return 0
    return int(prod(t.shape)) * t.element_size()


def _consumer_map(program: Program) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, nd in enumerate(program.nodes):
        for nm in nd.inputs:
            out.setdefault(nm, []).append(i)
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
