"""Structural (forward-shaped) transforms.

The universal read-mode / fused-store transforms live in `pto_fuser.transform`; the
region-driven contraction+epilogue generator lives in `pto_fuser.template`. This
package holds the **resident-state scan** rewrites — the recurrence family, which is
not a contraction+epilogue template: a pure `Program -> Program` rewrite parameterized
by the forward's dims, matching the canonical (all-staged) scan the builders emit.
"""
from .gdn import LowerResidentScan
from .kda import LowerPerDimScan

__all__ = ["LowerResidentScan", "LowerPerDimScan"]
