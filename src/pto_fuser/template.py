"""Template emission — region-driven fusion of a contraction + its epilogue.

The second megakernel step (``docs/DESIGN.md`` §8). Where the B0 analysis
(`analysis.py`) *finds* the fusible scopes, this *emits* the kernel: for each
contraction, it extracts the local **epilogue unit** (the einsum + its downstream
single-consumer glue chain + any per-operand prologue), matches it against a registry
of **proven** kernel templates, and rewrites the unit into that template's
`FusedNode`. One generator (`FuseContractionEpilogue`) replaces the three bespoke
epilogue-fusion transforms; adding a fused pattern is now adding a `Template`, and the
generator **never emits a pattern without a proven kernel behind it**.

Two deliberate boundaries, both evidence-driven:

* **Proven templates only.** The registry hosts the kernels we hand-proved on device
  (``kkt_gated_native`` / ``gated_qk_native`` — matmul-core + on-chip Vec/mask
  epilogue; ``qk_prologue`` — per-dim operand prescale + matmul). A contraction whose
  epilogue matches none of these is left staged, and `epilogue_report` says why.

* **No fixpipe-1D kernel yet — by evidence, not oversight.** §8 named the per-channel
  (1D) fixpipe-scale epilogue as the "hardware-clean" class to generate first, but the
  GDN/KDA epilogues are 2D score masks and per-row scales, which the *Vec*-epilogue
  templates above already cover; **no forward stage is a pure fixpipe-1D scale**, so
  such a kernel would have no consumer. It is the natural next template — to be
  hand-proved when a consumer appears (or exposed as a `pto-einsum` core capability) —
  not built speculatively here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import (EinsumNode, FusedNode, Program, TensorOp, VecGlueNode)
from .transform import Transform, TransformResult, producer_index, splice


# --------------------------------------------------------------------------- #
#  epilogue unit extraction
# --------------------------------------------------------------------------- #
@dataclass
class EpilogueUnit:
    """A contraction and the local glue a fused kernel could absorb around it.

    ``epilogue`` is the downstream single-consumer chain (mul / tril / contiguous)
    from the einsum to the unit's boundary output; ``prologue`` maps each anchor
    operand to the ``(idx, [base, coef])`` of the per-operand ``mul`` that produced it
    (a prescale that folds into the matmul load). ``boundary_out`` is the tensor the
    surrounding graph consumes."""
    anchor_idx: int
    anchor: EinsumNode
    epilogue: List[Tuple[int, object]]
    prologue: Dict[str, Tuple[int, List[str]]]
    boundary_out: str

    @property
    def epilogue_ops(self) -> List[str]:
        return [n.op for _, n in self.epilogue]


def _single_consumer(program: Program, name: str) -> Optional[int]:
    hits = [i for i, n in enumerate(program.nodes) if name in n.inputs]
    return hits[0] if len(hits) == 1 else None


def extract_epilogue_unit(program: Program, anchor_idx: int) -> EpilogueUnit:
    """Grow the epilogue chain + detect the prologue around the einsum at
    ``anchor_idx``. Pure structural walk — no shapes, no device."""
    anchor = program.nodes[anchor_idx]
    prod = producer_index(program)

    epilogue: List[Tuple[int, object]] = []
    cur = anchor.output
    while True:
        c = _single_consumer(program, cur)
        if c is None:
            break
        nd = program.nodes[c]
        if isinstance(nd, VecGlueNode) and nd.op in ("mul", "scale", "tril"):
            epilogue.append((c, nd)); cur = nd.output
        elif isinstance(nd, TensorOp) and nd.op == "contiguous":
            epilogue.append((c, nd)); cur = nd.output
        else:
            break

    prologue: Dict[str, Tuple[int, List[str]]] = {}
    for operand in anchor.inputs:
        p = prod.get(operand)
        if p is not None and isinstance(program.nodes[p], VecGlueNode) \
                and program.nodes[p].op == "mul":
            prologue[operand] = (p, list(program.nodes[p].inputs))
    return EpilogueUnit(anchor_idx, anchor, epilogue, prologue, cur)


# --------------------------------------------------------------------------- #
#  templates (each backed by a proven device kernel)
# --------------------------------------------------------------------------- #
@dataclass
class Dims:
    B: int
    H: int
    nc: int
    C: int
    D: int
    v2: Optional[bool] = None

    @property
    def N(self) -> int:
        return self.B * self.H


class Template:
    """A proven contraction+epilogue kernel, emittable from an `EpilogueUnit`."""
    name: str = "template"
    family: str = ""

    def matches(self, program: Program, unit: EpilogueUnit) -> bool:   # pragma: no cover
        raise NotImplementedError

    def emit(self, program: Program, unit: EpilogueUnit, dims: Dims
             ) -> Tuple[FusedNode, set]:   # (node, drop indices)   # pragma: no cover
        raise NotImplementedError


def _glue_coef(unit: EpilogueUnit, op: str) -> Optional[str]:
    """The non-anchor operand of the first ``op`` glue in the epilogue (e.g. the coef
    of a ``mul``)."""
    for _, n in unit.epilogue:
        if isinstance(n, VecGlueNode) and n.op == op:
            others = [x for x in n.inputs if x != unit.anchor.output]
            return others[0] if others else None
    return None


class GatedKKTTemplate(Template):
    """kkt: ``A = tril((k·kᵀ)⊙coefA, -1)`` -> ``kkt_gated_native`` (qk never in HBM)."""
    name = "kkt-gated-native"
    family = "scalar-gate"

    def matches(self, program, unit) -> bool:
        a, b = unit.anchor.inputs
        return (unit.anchor.equation == "nid,njd->nij" and a == b == "kF"
                and unit.epilogue_ops == ["mul", "tril", "contiguous"])

    def emit(self, program, unit, dims):
        kernel = "kkt_gated_native_v2" if dims.v2 else "kkt_gated_native"
        node = FusedNode(kernel=kernel, inputs=["kF", "g_native", "beta_native"],
                         outputs=[unit.boundary_out], params={"nc": dims.nc, "H": dims.H},
                         subsumes=["kkt einsum + coefA mul + tril (qk never in HBM)"])
        drop = {unit.anchor_idx} | {i for i, _ in unit.epilogue}
        return node, drop


class GatedChunkOTemplate(Template):
    """chunk_o intra: ``Aqk = (q·kᵀ)⊙coef_o`` -> ``gated_qk_native`` (causal, β=1)."""
    name = "chunk-o-gated-native"
    family = "scalar-gate"

    def matches(self, program, unit) -> bool:
        a, b = unit.anchor.inputs
        return (unit.anchor.equation == "nid,njd->nij" and a == "qF" and b == "kF"
                and unit.epilogue_ops == ["mul", "contiguous"]
                and _glue_coef(unit, "mul") == "coef_o")

    def emit(self, program, unit, dims):
        kernel = "gated_qk_native_v2" if dims.v2 else "gated_qk_native"
        node = FusedNode(kernel=kernel,
                         inputs=["qF", "kF", "g_native", "beta_native_ones"],
                         outputs=[unit.boundary_out],
                         params={"nc": dims.nc, "H": dims.H, "causal": True},
                         subsumes=["chunk_o Aqk einsum + coef_o mul + contiguous"])
        drop = {unit.anchor_idx} | {i for i, _ in unit.epilogue}
        return node, drop


class PerDimPrologueTemplate(Template):
    """KDA chunk_o: per-dim gate folded into the matmul load -> ``qk_prologue``.
    Matches ``q_eff=mul(qF,coef_ag)``, ``k_eff=mul(kF,coef_bg)`` feeding the einsum,
    then ``tril -> contiguous``. ``q_eff`` is left in place (it also feeds o_inter)."""
    name = "qk-prologue"
    family = "per-dim-prologue"

    def matches(self, program, unit) -> bool:
        a, b = unit.anchor.inputs
        pa, pb = unit.prologue.get(a), unit.prologue.get(b)
        if not (pa and pb):
            return False
        return (unit.anchor.equation == "nid,njd->nij"
                and pa[1] == ["qF", "coef_ag"] and pb[1] == ["kF", "coef_bg"]
                and unit.epilogue_ops == ["tril", "contiguous"])

    def emit(self, program, unit, dims):
        kernel = _select_prologue_kernel(dims.C, dims.D) if dims.v2 is None \
            else "qk_prologue_v2" if dims.v2 else "qk_prologue"
        node = FusedNode(kernel=kernel, inputs=["qF", "kF", "coef_ag", "coef_bg"],
                         outputs=[unit.boundary_out],
                         params={"nc": dims.nc, "H": dims.N, "C": dims.C, "D": dims.D},
                         subsumes=["chunk_o k_eff mul + Aqk einsum + tril + contiguous"])
        # k_eff (b's prologue mul) is exclusive to this unit -> drop it; q_eff is shared.
        b = unit.anchor.inputs[1]
        drop = {unit.anchor_idx} | {i for i, _ in unit.epilogue}
        kmul = unit.prologue[b][0]
        if _single_consumer(program, program.nodes[kmul].output) == unit.anchor_idx:
            drop.add(kmul)
        return node, drop


def _select_prologue_kernel(C: int, D: int) -> str:
    return "qk_prologue_v2" if C * D >= 4096 else "qk_prologue"


def default_templates() -> List[Template]:
    """The proven contraction+epilogue kernels, in match-priority order."""
    return [GatedKKTTemplate(), GatedChunkOTemplate(), PerDimPrologueTemplate()]


# --------------------------------------------------------------------------- #
#  the generator
# --------------------------------------------------------------------------- #
class FuseContractionEpilogue(Transform):
    """Emit every contraction+epilogue for which a proven template matches, replacing
    the einsum + its glue chain with the template's `FusedNode`. Region-driven and
    proven-only: a contraction with no matching template is left staged."""

    name = "fuse-contraction-epilogue"
    summary = "contraction + epilogue glue -> proven gated-matmul FusedNode (template)"

    def __init__(self, B, H, nc, C, D, v2: Optional[bool] = None,
                 templates: Optional[List[Template]] = None) -> None:
        self.dims = Dims(B, H, nc, C, D, v2)
        self.templates = templates if templates is not None else default_templates()

    def _matches(self, program: Program) -> List[Tuple[int, Template]]:
        """(anchor_idx, template) for each einsum whose epilogue a template matches."""
        out = []
        for i, n in enumerate(program.nodes):
            if not isinstance(n, EinsumNode):
                continue
            unit = extract_epilogue_unit(program, i)
            for t in self.templates:
                if t.matches(program, unit):
                    out.append((i, t))
                    break
        return out

    def match(self, program: Program) -> int:
        return len(self._matches(program))

    def apply(self, program: Program) -> TransformResult:
        fired = []
        prog = program
        # Re-extract per rewrite: indices shift after each splice, so recompute.
        while True:
            hits = self._matches(prog)
            if not hits:
                break
            i, t = hits[0]
            unit = extract_epilogue_unit(prog, i)
            node, drop = t.emit(prog, unit, self.dims)
            prog = splice(prog, drop, min(drop), [node])
            fired.append(t.name)
        note = "emitted: " + (", ".join(fired) if fired else "none")
        return TransformResult(prog, bool(fired), len(fired), note)


# --------------------------------------------------------------------------- #
#  chunk_o flash — score→output fused (B3), a STANDALONE transform
# --------------------------------------------------------------------------- #
class FuseChunkOFlash(Transform):
    """Fuse the whole scalar-gate chunk_o score→output — ``q·kᵀ → gate/mask → ·v`` —
    into ONE ``qkv_flash_native`` kernel, so the [M,C,C] masked score never lands in
    HBM. Matches the canonical Aqk einsum + ``coef_o`` mul + contiguous + the o_intra
    ``nij,nje->nie`` contraction and replaces all four with the flash FusedNode
    (recomputing the exp gate + causal mask on-chip from g_native, so it reads
    (qF, kF, v, g_native, beta_native_ones) — not the score).

    It is a **separate** transform from `FuseContractionEpilogue`, and deliberately so:
    under graph capture the o_intra dispatch is already elided and flash-V1 trades the
    A round-trip for a larger S round-trip, so it is a *dispatch-regime* win (≈5× on the
    un-captured / dynamic-shape path) but ≈parity-to-regression captured. Keeping it a
    standalone, independently-verified transform means the propose/verify/dispose loop
    drops it wherever it does not pay **without** disturbing the kkt fusion — which was
    exactly the failure of bundling it into the epilogue generator. Ordered before
    `fuse-contraction-epilogue`: if kept, chunk_o is already flashed and the generator
    emits only kkt; if dropped, the generator emits the score-only chunk_o kernel."""

    name = "fuse-chunk-o-flash"
    summary = "chunk_o q·kᵀ → gate/mask → ·v -> one qkv_flash_native (score never in HBM)"

    def __init__(self, B, H, nc, C, D) -> None:
        self.dims = Dims(B, H, nc, C, D)

    def _site(self, program: Program) -> Optional[Tuple[int, EpilogueUnit, int]]:
        """(anchor_idx, unit, o_intra_idx) for the chunk_o score→output chain, else None."""
        for i, n in enumerate(program.nodes):
            if not (isinstance(n, EinsumNode) and n.equation == "nid,njd->nij"
                    and n.inputs == ["qF", "kF"]):
                continue
            unit = extract_epilogue_unit(program, i)
            if not (unit.epilogue_ops == ["mul", "contiguous"]
                    and _glue_coef(unit, "mul") == "coef_o"):
                continue
            c = _single_consumer(program, unit.boundary_out)
            if c is None:
                continue
            cn = program.nodes[c]
            if (isinstance(cn, EinsumNode) and cn.equation == "nij,nje->nie"
                    and cn.inputs[0] == unit.boundary_out):
                return i, unit, c
        return None

    def match(self, program: Program) -> int:
        return 1 if self._site(program) is not None else 0

    def apply(self, program: Program) -> TransformResult:
        site = self._site(program)
        if site is None:
            return TransformResult(program, False, 0, "no chunk_o score→output chain")
        i, unit, c = site
        o_intra = program.nodes[c]
        v_name = o_intra.inputs[1]
        # V2 (double-buffered interleave) keeps S+A L2-resident and hides the Cube↔Vec
        # handshake — the variant that beats the captured staged path (V1 pays an S
        # round-trip). The verifier still gates it; a slower shape falls back to canonical.
        node = FusedNode(kernel="qkv_flash_native_v2",
                         inputs=["qF", "kF", v_name, "g_native", "beta_native_ones"],
                         outputs=[o_intra.output],
                         params={"nc": self.dims.nc, "H": self.dims.H, "C": self.dims.C,
                                 "D": self.dims.D, "DV": self.dims.D, "causal": True},
                         subsumes=["chunk_o Aqk einsum + coef_o mul + contiguous + "
                                   "o_intra A@v (masked score never in HBM)"])
        drop = {i} | {j for j, _ in unit.epilogue} | {c}
        prog = splice(program, drop, min(drop), [node])
        return TransformResult(prog, True, 1, "emitted qkv_flash_native")


class FusePerDimChunkOFlash(Transform):
    """Per-dim (KDA/GLA) chunk_o flash — the per-channel-gate twin of `FuseChunkOFlash`.
    Matches the canonical per-dim score→output chain: ``q_eff=mul(qF,coef_ag)`` and
    ``k_eff=mul(kF,coef_bg)`` feeding the Aqk einsum → ``tril`` → contiguous → the
    o_intra ``nij,nje->nie`` contraction, and replaces it with ONE ``qkvp_flash_native_v2``
    kernel (Vec prescale → Cube score → Vec tril → Cube A·v, ops+S+A L2-resident).
    ``q_eff`` is kept (it also feeds o_inter); ``k_eff`` is dropped when exclusive. Reads
    (qF, kF, v, coef_ag, coef_bg) — the decay rides on the operands, not a scalar coeff.
    Standalone + independently verified, exactly like the scalar flash."""

    name = "fuse-perdim-chunk-o-flash"
    summary = "per-dim chunk_o prescale·q·kᵀ → tril → ·v -> one qkvp_flash_native_v2"

    def __init__(self, B, H, nc, C, D) -> None:
        self.dims = Dims(B, H, nc, C, D)

    def _site(self, program: Program) -> Optional[Tuple[int, EpilogueUnit, int]]:
        """(anchor_idx, unit, o_intra_idx) for the per-dim chunk_o chain, else None."""
        for i, n in enumerate(program.nodes):
            if not (isinstance(n, EinsumNode) and n.equation == "nid,njd->nij"):
                continue
            unit = extract_epilogue_unit(program, i)
            a, b = unit.anchor.inputs
            pa, pb = unit.prologue.get(a), unit.prologue.get(b)
            if not (pa and pb):
                continue
            if not (pa[1] == ["qF", "coef_ag"] and pb[1] == ["kF", "coef_bg"]
                    and unit.epilogue_ops == ["tril", "contiguous"]):
                continue
            c = _single_consumer(program, unit.boundary_out)
            if c is None:
                continue
            cn = program.nodes[c]
            if (isinstance(cn, EinsumNode) and cn.equation == "nij,nje->nie"
                    and cn.inputs[0] == unit.boundary_out):
                return i, unit, c
        return None

    def match(self, program: Program) -> int:
        return 1 if self._site(program) is not None else 0

    def apply(self, program: Program) -> TransformResult:
        site = self._site(program)
        if site is None:
            return TransformResult(program, False, 0, "no per-dim chunk_o score→output chain")
        i, unit, c = site
        o_intra = program.nodes[c]
        v_name = o_intra.inputs[1]
        node = FusedNode(kernel="qkvp_flash_native_v2",
                         inputs=["qF", "kF", v_name, "coef_ag", "coef_bg"],
                         outputs=[o_intra.output],
                         params={"nc": self.dims.nc, "H": self.dims.H, "C": self.dims.C,
                                 "D": self.dims.D, "DV": self.dims.D, "causal": True},
                         subsumes=["chunk_o k_eff mul + Aqk einsum + tril + contiguous + "
                                   "o_intra A@v (masked score never in HBM)"])
        drop = {i} | {j for j, _ in unit.epilogue} | {c}
        # k_eff (b's prescale mul) is exclusive to this unit -> drop it; q_eff feeds o_inter.
        b = unit.anchor.inputs[1]
        kmul = unit.prologue[b][0]
        if _single_consumer(program, program.nodes[kmul].output) == i:
            drop.add(kmul)
        prog = splice(program, drop, min(drop), [node])
        return TransformResult(prog, True, 1, "emitted qkvp_flash_native_v2")


# --------------------------------------------------------------------------- #
#  reporting (which epilogues are template-eligible; the fixpipe-1D finding)
# --------------------------------------------------------------------------- #
@dataclass
class EpilogueReport:
    rows: List[Tuple[str, str, str]] = field(default_factory=list)  # (anchor_out, epilogue, verdict)

    def __str__(self) -> str:
        head = "contraction-epilogue units (template = proven fused kernel):"
        lines = [head]
        for anchor, epi, verdict in self.rows:
            lines.append(f"  {anchor:<12} epilogue[{epi:<22}] -> {verdict}")
        matched = sum(v.startswith("template ") for _, _, v in self.rows)
        lines.append(f"  -- {matched}/{len(self.rows)} contractions match a proven template; "
                     f"the rest stay staged (per-row scale / plain matmul — no fixpipe-1D-scale "
                     f"stage exists to generate).")
        return "\n".join(lines)


def epilogue_report(program: Program, templates: Optional[List[Template]] = None
                    ) -> EpilogueReport:
    """Classify every contraction's epilogue: which proven template emits it, or why
    it stays staged. Pure structural analysis (no device)."""
    templates = templates if templates is not None else default_templates()
    rep = EpilogueReport()
    for i, n in enumerate(program.nodes):
        if not isinstance(n, EinsumNode):
            continue
        unit = extract_epilogue_unit(program, i)
        epi = "+".join(unit.epilogue_ops) or "(none)"
        verdict = "staged (no matching template)"
        for t in templates:
            if t.matches(program, unit):
                verdict = f"template {t.name}"
                break
        rep.rows.append((n.output, epi, verdict))
    return rep
