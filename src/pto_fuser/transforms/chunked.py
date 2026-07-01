"""Chunked-linear-attention transform — batch the per-chunk intra-scores (B2).

The linear family (vanilla LA, RetNet, Mamba-2, GLA) is one chunked recurrence with
different decay. Its canonical form unrolls the intra-chunk score
``A_c = tril(q̃_c·k̂_cᵀ, 0)`` per chunk *inside* the scan loop — ``nc`` tiny einsums.
`BatchChunkIntraScore` collapses all ``nc`` of them into ONE batched kernel over
``M = N·nc`` chunks, reusing the **proven** kernels the GDN/KDA templates already use:

* scalar gate (vanilla/RetNet/Mamba-2) — the decay factors *out* of the contraction,
  so the score is the same ``gated_qk_native_v2`` epilogue GDN's kkt uses (q/k raw +
  the scalar log-cumgate ``g_intra``);
* per-channel gate (GLA) — the decay rides *on* the operands, so a ``qk_prologue``
  kernel prescales ``q⊙P`` / ``k⊙(1/P)`` ahead of the matmul.

No new device code — this widens the *coverage* of the proven template kernels to the
whole linear family (B2: widen one proven pattern at a time). It is a forward-shaped
structural rewrite, byte-identical to the hand-written ``fused_intra`` lowering, and
the gate kind (scalar vs per-dim) is a forward-declared **option** — the one property
the canonical IR cannot carry, and a concrete signal for the eventual frontend.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..ir import EinsumNode, FusedNode, Program, TensorOp, VecGlueNode
from ..transform import Transform, TransformResult


def _select_prologue_kernel(C: int, d_k: int) -> str:
    return "qk_prologue_v2" if C * d_k >= 4096 else "qk_prologue"


class BatchChunkIntraScore(Transform):
    """Replace the ``nc`` per-chunk ``tril(einsum)`` intra-scores with one batched
    proven kernel over all chunks + a reshape + per-chunk slices."""

    name = "batch-chunk-intra-score"
    summary = "nc per-chunk tril(q̃·k̂ᵀ) intra-scores -> one batched gated/prologue kernel"

    def __init__(self, N: int, nc: int, C: int, d_k: int, d_v: int,
                 per_dim_gate: bool = False, v2: Optional[bool] = None) -> None:
        self.N, self.nc, self.C, self.d_k, self.d_v = N, nc, C, d_k, d_v
        self.per_dim_gate, self.v2 = per_dim_gate, v2

    def _intra_scores(self, program: Program) -> List[Tuple[int, int, str]]:
        """(einsum_idx, tril_idx, score_name) for each per-chunk intra score:
        ``einsum "nid,njd->nij" -> tril(0) -> o_intra einsum "nij,nje->nie"``."""
        hits = []
        for i, n in enumerate(program.nodes):
            if not (isinstance(n, EinsumNode) and n.equation == "nid,njd->nij"):
                continue
            t = _single_consumer(program, n.output)
            if t is None or not _is_glue(program.nodes[t], "tril"):
                continue
            o = _single_consumer(program, program.nodes[t].output)
            if o is None or not (isinstance(program.nodes[o], EinsumNode)
                                 and program.nodes[o].equation == "nij,nje->nie"):
                continue
            hits.append((i, t, program.nodes[t].output))
        return hits

    def match(self, program: Program) -> int:
        if any(isinstance(n, FusedNode) and n.output == "A_all" for n in program.nodes):
            return 0                                    # already fused
        return 1 if self._intra_scores(program) else 0

    def apply(self, program: Program) -> TransformResult:
        hits = self._intra_scores(program)
        if not hits:
            return TransformResult(program, False, 0, "no per-chunk intra scores")
        nc = self.nc
        if self.per_dim_gate:
            kernel = "qk_prologue_v2" if self.v2 else \
                "qk_prologue" if self.v2 is False else _select_prologue_kernel(self.C, self.d_k)
            a_all = FusedNode(kernel=kernel, inputs=["q", "k", "P", "invP"],
                              outputs=["A_all"],
                              params={"nc": nc, "H": self.N, "C": self.C, "D": self.d_k},
                              subsumes=[nm for _, _, nm in hits])
        else:
            a_all = FusedNode(kernel="gated_qk_native_v2",
                              inputs=["q", "k", "g_intra", "beta_intra"],
                              outputs=["A_all"],
                              params={"nc": nc, "H": self.N, "C": self.C,
                                      "D": self.d_k, "causal": True},
                              subsumes=[nm for _, _, nm in hits])
        a_allr = TensorOp("reshape", ["A_all"], "A_allr",
                          params={"shape": (self.N, nc, self.C, self.C)})

        drop = {i for i, _, _ in hits} | {t for _, t, _ in hits}
        slice_at = {t: c for c, (_, t, _) in enumerate(hits)}   # tril idx -> chunk index
        new_nodes = [a_all, a_allr]
        for idx, node in enumerate(program.nodes):
            if idx in drop:
                if idx in slice_at:                     # tril -> slice of A_allr
                    c = slice_at[idx]
                    new_nodes.append(TensorOp("slice", ["A_allr"], node.output,
                                              params={"axis": 1, "index": c}))
                continue                                # einsum: drop
            new_nodes.append(node)
        prog = Program(nodes=new_nodes, inputs=list(program.inputs),
                       outputs=list(program.outputs))
        return TransformResult(prog, True, 1,
                               f"{len(hits)} per-chunk intra scores -> {a_all.kernel}")


def _single_consumer(program: Program, name: str) -> Optional[int]:
    hits = [i for i, n in enumerate(program.nodes) if name in n.inputs]
    return hits[0] if len(hits) == 1 else None


def _is_glue(node, op: str) -> bool:
    return isinstance(node, VecGlueNode) and node.op == op
