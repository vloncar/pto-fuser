"""GDN structural transforms — resident-state scan and scalar-gated glue absorption.

Each matches the canonical (all-staged) GDN program `attention/_gdn_full` emits and
rewrites one region into a hosted `FusedNode`, keeping an intermediate on-chip that
the staged lowering round-trips HBM. They are pure IR rewrites parameterized by the
forward's dims (the pass "options"); the *decision* to apply them and the
correctness/perf *gate* live in the policy and the verifier, not here.

The canonical node names these match are the documented contract of the GDN builder
(the builder emits exactly the staged form; these transforms are its fused
lowerings, lifted out of the old build-flag branches).
"""
from __future__ import annotations

import torch

from ..ir import (EinsumNode, FusedNode, Program, TensorOp, VecGlueNode,
                  node_outputs)
from ..transform import Transform, TransformResult, producer_index


def _has_fused(program: Program, kernel_prefix: str) -> bool:
    return any(isinstance(n, FusedNode) and n.kernel.startswith(kernel_prefix)
               for n in program.nodes)


# --------------------------------------------------------------------------- #
#  resident-state scan (chunk_h_scan)
# --------------------------------------------------------------------------- #
class LowerResidentScan(Transform):
    """Replace the unrolled ``nc``-chunk cross-chunk scan (which writes the carried
    state ``S`` back to HBM every chunk) with the ``chunk_h_scan`` `FusedNode` that
    keeps ``S`` resident on-chip, plus the parallel ``v_new = U - W·h_out``
    recompute. Removes the per-chunk HBM round-trip of ``S`` — a bandwidth win graph
    capture cannot touch (it fuses broadly, not just launch-bound)."""

    name = "lower-resident-scan"
    summary = "unrolled chunk_h scan -> chunk_h_scan FusedNode (resident S)"

    def __init__(self, B: int, H: int, nc: int, C: int, D: int,
                 work: torch.dtype = torch.float16) -> None:
        self.B, self.H, self.nc, self.C, self.D = B, H, nc, C, D
        self.M = B * H * nc
        self.work = work

    def _region(self, program: Program):
        """Return (drop_indices, insert_at) for the canonical scan, or None."""
        if _has_fused(program, "chunk_h_scan"):
            return None
        if "coef_vcs" not in program.inputs:       # GDN scalar-decay scan only
            return None
        prod = producer_index(program)
        if "S0" not in prod or "h_flat" not in prod or "vn_flat" not in prod:
            return None
        names = {"S0", "h_bh", "h_flat", "vn_bh", "vn_flat"}
        for c in range(self.nc):
            names.update({f"Wc{c}_s", f"Wc{c}", f"WS{c}", f"Uc{c}", f"vn{c}",
                          f"vcs{c}", f"vn2{c}", f"kc{c}_s", f"kc{c}", f"dS{c}",
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
        new = [
            VecGlueNode("mul", ["kb5", "coef_vcs"], "k_vcs", out_dtype=work),
        ]
        for nm, src in (("sw", "Wb"), ("su", "Ub"), ("sk", "k_vcs")):
            new.append(TensorOp("reshape", [src], f"{nm}5",
                                params={"shape": [B, H, nc, C, D]}))
            new.append(TensorOp("permute", [f"{nm}5"], nm,
                                params={"dims": (0, 2, 3, 1, 4)}))
        new.append(TensorOp("reshape", ["coef_S"], "sdecay",
                            params={"shape": [B, H, nc]}))
        new.append(FusedNode(kernel="chunk_h_scan",
                             inputs=["sw", "su", "sk", "sdecay"],
                             outputs=["h_out_k", "final_k"],
                             params={"B": B, "H": H, "nc": nc},
                             subsumes=["nc per-chunk WS/kv matmul pairs + residual glue"]))
        new.append(TensorOp("permute", ["h_out_k"], "h_bhn",
                            params={"dims": (0, 2, 1, 3, 4)}))
        new.append(TensorOp("reshape", ["h_bhn"], "h_flat",
                            params={"shape": [M, D, D]}))
        new.append(EinsumNode("nid,nde->nie", ["W_m", "h_flat"], "WS_all",
                              out_dtype=work, read_mode="NN", fuse_out=False))
        new.append(VecGlueNode("sub", ["U_m", "WS_all"], "vn_flat", out_dtype=work))

        from ..transform import splice
        prog = splice(program, drop, insert_at, new)
        return TransformResult(prog, True, 1,
                               f"scan of {nc} chunks -> chunk_h_scan resident state")

