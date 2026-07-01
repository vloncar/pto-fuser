"""KDA structural transforms — the per-channel analogs of the GDN fusions.

KDA shares GDN's four contractions but its gate is **per-dimension**: it is baked
into the operands (``exp(±g)``) rather than a scalar score factor. So the scan uses a
per-dim decay vector (``chunk_h_scan`` with ``perdim_decay=True``). The chunk_o
glue-absorption (a `qk_prologue` `FusedNode`) is emitted by the region-driven
`template.FuseContractionEpilogue` generator (its `PerDimPrologueTemplate`), not a
bespoke transform here; kkt stays a plain einsum for KDA.
"""
from __future__ import annotations

import torch

from ..ir import (EinsumNode, FusedNode, Program, TensorOp, VecGlueNode,
                  node_outputs)
from ..transform import Transform, TransformResult, producer_index, splice
from .gdn import _has_fused


class LowerPerDimScan(Transform):
    """KDA resident-state scan: same rewrite as GDN's `LowerResidentScan` but with a
    **per-dimension** cross-chunk decay — ``chunk_h_scan`` built with
    ``perdim_decay=True`` and an ``[B,H,nc,D]`` decay operand."""

    name = "lower-perdim-scan"
    summary = "unrolled KDA scan -> chunk_h_scan FusedNode (resident S, per-dim decay)"

    def __init__(self, B: int, H: int, nc: int, C: int, D: int,
                 work: torch.dtype = torch.float16) -> None:
        self.B, self.H, self.nc, self.C, self.D = B, H, nc, C, D
        self.M = B * H * nc
        self.work = work

    def _region(self, program: Program):
        if _has_fused(program, "chunk_h_scan"):
            return None
        if "coef_krest" not in program.inputs:     # KDA per-dim-decay scan only
            return None
        prod = producer_index(program)
        if "S0" not in prod or "h_flat" not in prod or "vn_flat" not in prod:
            return None
        names = {"S0", "h_bh", "h_flat", "vn_bh", "vn_flat"}
        for c in range(self.nc):
            names.update({f"Wc{c}_s", f"Wc{c}", f"WS{c}", f"Uc{c}", f"vn{c}",
                          f"kc{c}_s", f"kc{c}", f"krc{c}", f"krest{c}", f"dS{c}",
                          f"sc{c}", f"Sd{c}", f"S{c + 1}"})
        drop = {i for i, n in enumerate(program.nodes)
                if any(o in names for o in node_outputs(n))}
        if not drop:
            return None
        return drop, min(drop)

    def match(self, program: Program) -> int:
        return 1 if self._region(program) is not None else 0

    def apply(self, program: Program) -> TransformResult:
        region = self._region(program)
        if region is None:
            return TransformResult(program, False, 0, "no canonical scan region")
        drop, insert_at = region
        B, H, nc, C, D, M, work = (self.B, self.H, self.nc, self.C, self.D,
                                   self.M, self.work)
        new = [VecGlueNode("mul", ["kb5", "coef_krest"], "k_krest", out_dtype=work)]
        for nm, src in (("sw", "Wb"), ("su", "Ub"), ("sk", "k_krest")):
            new.append(TensorOp("reshape", [src], f"{nm}5",
                                params={"shape": [B, H, nc, C, D]}))
            new.append(TensorOp("permute", [f"{nm}5"], nm,
                                params={"dims": (0, 2, 3, 1, 4)}))
        new.append(TensorOp("reshape", ["coef_S"], "sdecay",
                            params={"shape": [B, H, nc, D]}))
        new.append(FusedNode(kernel="chunk_h_scan",
                             inputs=["sw", "su", "sk", "sdecay"],
                             outputs=["h_out_k", "final_k"],
                             params={"B": B, "H": H, "nc": nc, "perdim_decay": True},
                             subsumes=["nc per-chunk WS/kv matmul pairs + residual glue"]))
        new.append(TensorOp("permute", ["h_out_k"], "h_bhn",
                            params={"dims": (0, 2, 1, 3, 4)}))
        new.append(TensorOp("reshape", ["h_bhn"], "h_flat",
                            params={"shape": [M, D, D]}))
        new.append(EinsumNode("nid,nde->nie", ["W_m", "h_flat"], "WS_all",
                              out_dtype=work, read_mode="NN", fuse_out=False))
        new.append(VecGlueNode("sub", ["U_m", "WS_all"], "vn_flat", out_dtype=work))
        prog = splice(program, drop, insert_at, new)
        return TransformResult(prog, True, 1,
                               f"per-dim scan of {nc} chunks -> chunk_h_scan resident state")

