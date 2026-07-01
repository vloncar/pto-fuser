"""Compilation report — the provenance the propose/verify/dispose loop leaves behind.

Every transform the policy proposed produces one `TransformRecord`: what the cost
model predicted, whether it structurally matched, and — if verified on device — the
measured frob / determinism / speedup that decided its fate. The report is both the
human-readable trace of *why the lowered program looks the way it does* and the
data that recalibrates the cost model (predicted benefit vs measured speedup).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TransformRecord:
    name: str
    summary: str
    predicted_benefit: float
    predicted_rationale: str
    matched: bool                       # structurally applicable when reached
    attempted: bool                     # applied (verified or unverified)
    kept: bool                          # survives in the lowered program
    # verification measurements (None when unverified / not reached)
    frob: Optional[float] = None
    deterministic: Optional[bool] = None
    faster: Optional[bool] = None
    t_base_ms: Optional[float] = None
    t_cand_ms: Optional[float] = None
    note: str = ""

    @property
    def speedup(self) -> Optional[float]:
        if self.t_base_ms and self.t_cand_ms:
            return self.t_base_ms / self.t_cand_ms
        return None

    def __str__(self) -> str:
        verdict = "KEEP" if self.kept else ("drop" if self.matched else "n/a ")
        pred = f"pred {self.predicted_benefit:+.2f}"
        if self.speedup is not None:
            meas = (f"{self.t_base_ms:.3f}->{self.t_cand_ms:.3f}ms "
                    f"({self.speedup:.2f}×)")
            flags = f"frob={self.frob:.1e} {'det' if self.deterministic else 'NDET'}"
        else:
            meas, flags = ("(unverified)" if self.attempted else self.note), ""
        return f"[{verdict}] {self.name:<22} {pred:<11} {flags:<20} {meas}"


@dataclass
class CompilationReport:
    features: object
    records: List[TransformRecord] = field(default_factory=list)
    verified: bool = False

    @property
    def kept(self) -> List[TransformRecord]:
        return [r for r in self.records if r.kept]

    def __str__(self) -> str:
        head = (f"compile report ({'verified' if self.verified else 'unverified'}) "
                f"— {len(self.kept)}/{len(self.records)} transforms kept")
        return "\n".join([head] + [str(r) for r in self.records])
