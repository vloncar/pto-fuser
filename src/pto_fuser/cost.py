"""Cost model — the *heuristic*, separated from the transforms and the verifier.

A `Transform` says *what* rewrite is possible; the cost model predicts *whether it
is likely to pay*, from the problem's shape/dtype features. It does **not** decide
alone — the verifier (`compile.py`) measures every proposed transform and can
override a prediction (a mispredicted win is a perf event, never a correctness one,
because the canonical lowering is always the floor). The model exists so the policy
can *order* and *prune* the pipeline, and so the compilation report can show
predicted-vs-measured — the signal that recalibrates these heuristics over time.

The predictions are seeded from the measured GDN scaling grid (the
`gdn-scaling-heads-x-seqlen` characterization): the fuser is launch/dispatch-bound at
small head count and bandwidth-bound at large head count, and the fuser-wins crossover
context length *grows as heads shrink*. That regime split is what the benefit
estimates below encode; they are deliberately coarse (rank, not absolute time) and are
meant to be refined against `CompilationReport` data, not trusted as ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# --------------------------------------------------------------------------- #
#  features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Features:
    """The shape/dtype signature a cost prediction keys on. These are the dims a
    chunked linear-attention forward is parameterized by; the transforms take the
    same dims as their options, so a `Features` both drives the cost model and
    constructs the structural transforms."""
    B: int
    H: int
    nc: int
    C: int
    D: int
    dtype: torch.dtype = torch.float16

    @property
    def T(self) -> int:
        return self.nc * self.C

    @property
    def M(self) -> int:               # parallel-stage batch (chunk-independent)
        return self.B * self.H * self.nc

    @property
    def N(self) -> int:               # scan chains (one per (b,h))
        return self.B * self.H

    @property
    def regime(self) -> str:
        """Coarse launch-vs-bandwidth classification from the measured grid: small
        head count is launch/dispatch-bound (dispatch-elim + resident state dominate),
        large head count is bandwidth-bound (glue absorption matters most), with a
        context-dependent crossover between."""
        if self.H <= 2:
            return "launch-bound"
        if self.H >= 16:
            return "bandwidth-bound"
        return "crossover"


@dataclass
class Prediction:
    """A cost-model verdict for one transform on one `Features`: a coarse benefit
    rank (>0 = expected win, ≤0 = expected no-op/loss) plus the rationale, and an
    optional ``v2`` hint for the shape-gated fused kernels."""
    benefit: float
    rationale: str
    v2: Optional[bool] = None

    @property
    def worth_trying(self) -> bool:
        return self.benefit > 0


# --------------------------------------------------------------------------- #
#  the model
# --------------------------------------------------------------------------- #
class CostModel:
    """Seeded predictor over the transform library. Coarse by design — the verifier
    is the ground truth; this only ranks/prunes and feeds the calibration loop."""

    def predict(self, transform_name: str, feat: Features) -> Prediction:
        name = transform_name
        if name == "enable-direct-reads":
            # Huge on the head-strided GDN/KDA family (h is a non-innermost batch
            # axis → Phase-A pays a strided gather the direct read removes: measured
            # 2.7–10.3×); ~1.0× on flat [M,C,D] families. Cheap to verify, so always
            # worth trying; benefit ranked by how strided the batch is (head count).
            return Prediction(1.0 + 0.02 * feat.H,
                              "direct read removes Phase-A strided gather on the head axis")
        if name == "enable-fused-store":
            return Prediction(0.5, "fused permuted store drops Phase C when free1-innermost")
        if name in ("lower-resident-scan", "lower-perdim-scan"):
            # Removes the per-chunk HBM round-trip of S — a bandwidth win that grows
            # with chunk count and fuses broadly (not just launch-bound: 4.40× at
            # B1H4nc8, 2.28× at B8H32nc16).
            return Prediction(1.0 + 0.05 * feat.nc,
                              f"resident state removes {feat.nc} per-chunk S round-trips")
        if name == "fuse-contraction-epilogue":
            # Region-driven glue absorption (contraction + epilogue -> one gated-matmul
            # kernel): keeps the qk matrix on-chip; the glue share grows with head count
            # (large-H captured forward is glue-bound), so the win rises with H. v2
            # (L2-resident, no [M,C,C] round-trip) pays where the score/head product is
            # large.
            v2 = feat.C * feat.D >= 4096
            return Prediction(0.3 + 0.03 * feat.H,
                              "glue absorption keeps qk on-chip (glue-bound at large H)",
                              v2=v2)
        return Prediction(0.0, "unknown transform — no prediction")
