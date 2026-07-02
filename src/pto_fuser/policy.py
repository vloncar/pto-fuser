"""Policy — turns a cost model + a program into an ordered transform pipeline.

The policy is the one place that *chooses*: given a canonical program and its
`Features`, it instantiates the candidate transforms (parameterized by the dims),
drops the ones that do not structurally match, asks the cost model whether each is
worth trying, and orders the survivors (structural fusions first — they remove
einsums — then the read-mode / fused-store annotation levers over whatever einsums
remain). It returns *proposals*, not decisions: the compile driver verifies each on
device and keeps it only on a measured, gated win.

The candidate set is the whole transform library; ``match`` prunes it to the ones
that apply to *this* program (GDN vs KDA vs a flat family), so the policy stays
forward-agnostic — adding a forward means adding transforms, not editing the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .cost import CostModel, Features, Prediction
from .ir import Program
from .transform import EnableDirectReads, EnableFusedStore, Transform


# structural fusions before annotation levers (fusions remove einsums the
# annotation levers would otherwise measure and then find gone).
_ORDER = {
    "lower-resident-scan": 0, "lower-perdim-scan": 0,
    "fuse-chunk-o-flash": 1,                                  # before the epilogue generator
    "fuse-contraction-epilogue": 2, "batch-chunk-intra-score": 2,
    "enable-direct-reads": 3, "enable-fused-store": 4,
}


@dataclass
class PlannedTransform:
    """One proposed transform plus the cost-model prediction that proposed it."""
    transform: Transform
    prediction: Prediction

    @property
    def name(self) -> str:
        return self.transform.name


class Policy:
    """Default policy: propose every applicable transform the cost model expects to
    pay, ordered fusions-first. Injectable cost model for testing / recalibration."""

    def __init__(self, cost: Optional[CostModel] = None) -> None:
        self.cost = cost or CostModel()

    def candidates(self, feat: Features) -> List[Transform]:
        """The full transform library, parameterized by ``feat``. ``match`` prunes
        to those that apply to the program at hand."""
        from .transforms import (BatchChunkIntraScore, LowerPerDimScan,
                                  LowerResidentScan)
        from .template import FuseContractionEpilogue, FuseChunkOFlash
        B, H, nc, C, D = feat.B, feat.H, feat.nc, feat.C, feat.D
        v2 = self.cost.predict("fuse-contraction-epilogue", feat).v2
        intra_v2 = self.cost.predict("batch-chunk-intra-score", feat).v2
        return [
            LowerResidentScan(B, H, nc, C, D, feat.dtype),
            LowerPerDimScan(B, H, nc, C, D, feat.dtype),
            FuseChunkOFlash(B, H, nc, C, D),
            FuseContractionEpilogue(B, H, nc, C, D, v2=v2),
            BatchChunkIntraScore(feat.N, nc, C, D, D,
                                 per_dim_gate=feat.per_dim_gate,
                                 v2=intra_v2 if feat.per_dim_gate else None),
            EnableDirectReads(),
            EnableFusedStore(),
        ]

    def pipeline(self, program: Program, feat: Features) -> List[PlannedTransform]:
        planned: List[PlannedTransform] = []
        for t in self.candidates(feat):
            if t.match(program) == 0:
                continue
            pred = self.cost.predict(t.name, feat)
            if not pred.worth_trying:
                continue
            planned.append(PlannedTransform(t, pred))
        planned.sort(key=lambda p: _ORDER.get(p.name, 99))
        return planned
