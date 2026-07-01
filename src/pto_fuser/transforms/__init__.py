"""Structural (forward-shaped) transforms.

The universal read-mode / fused-store transforms live in `pto_fuser.transform`;
this package holds the transforms that match a *structural pattern* in a chunked
recurrence and rewrite it into a hosted fused kernel — resident-state scan and
glue absorption. Each is a pure `Program -> Program` rewrite parameterized by the
forward's dims (the way a `cce-mlir` pass is parameterized by its options), matching
the canonical (all-staged) program the builders emit and the verifier gates against.
"""
from .gdn import (AbsorbGatedChunkO, AbsorbGatedKKT, LowerResidentScan)
from .kda import (AbsorbQKPrologue, LowerPerDimScan)

__all__ = [
    "LowerResidentScan", "AbsorbGatedKKT", "AbsorbGatedChunkO",
    "LowerPerDimScan", "AbsorbQKPrologue",
]
