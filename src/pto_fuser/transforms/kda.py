"""KDA structural transforms — the per-channel analogs of the GDN fusions.

KDA shares GDN's four contractions but its gate is **per-dimension**: it is baked
into the operands (``exp(±g)``) rather than a scalar score factor. So the scan uses a
per-dim decay vector (``chunk_h_scan`` with ``perdim_decay=True``) and chunk_o folds
the decay into the matmul *load* (a `qk_prologue` `FusedNode`, not the scalar-gate
epilogue). kkt stays a plain einsum (no gated-kkt fusion for KDA).
"""
from __future__ import annotations

import torch

from ..ir import (EinsumNode, FusedNode, Program, TensorOp, VecGlueNode,
                  node_outputs)
from ..transform import Transform, TransformResult, producer_index, splice
from .gdn import _has_fused, _is_glue, _is_tensorop, _single_consumer


def _select_prologue_kernel(C: int, D: int) -> str:
    """Shape-gate the per-dim prologue lowering (mirrors the examples' selector):
    the L2-ring V2 wins where the prescale is bandwidth-heavy (``C·D ≥ 4096``); V1's
    batched two-pass wins at tiny tiles. ``fusion.decide`` overrides by measurement."""
    return "qk_prologue_v2" if C * D >= 4096 else "qk_prologue"


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


class AbsorbQKPrologue(Transform):
    """KDA chunk_o: fold the per-dim gate into the matmul load. Match
    ``k_eff=mul(kF,coef_bg) -> Aqk=(q_eff·k_effᵀ) -> tril -> contiguous`` and replace
    with a `qk_prologue` `FusedNode` (``q⊙exp(g)`` / ``k⊙exp(-g)`` prescale in the
    load, causal mask in the kernel). ``q_eff`` is left in place — it also feeds
    ``o_inter`` (``q·h``)."""

    name = "absorb-qk-prologue"
    summary = "KDA chunk_o k_eff mul + Aqk einsum + tril + contiguous -> qk_prologue FusedNode"

    def __init__(self, B: int, H: int, nc: int, C: int, D: int,
                 v2: bool = None) -> None:
        self.N = B * H
        self.nc, self.C, self.D, self.v2 = nc, C, D, v2

    def _region(self, program: Program):
        prod = producer_index(program)
        for i, n in enumerate(program.nodes):
            if not (isinstance(n, EinsumNode) and n.equation == "nid,njd->nij"):
                continue
            a, b = n.inputs               # q_eff, k_eff
            kp = prod.get(b)
            if kp is None:
                continue
            keff = program.nodes[kp]
            if not (_is_glue(keff, "mul") and list(keff.inputs) == ["kF", "coef_bg"]):
                continue
            if _single_consumer(program, b) != i:      # k_eff feeds only this einsum
                continue
            chain = [kp, i]
            cur = n.output
            t_i = _single_consumer(program, cur)
            if t_i is None or not _is_glue(program.nodes[t_i], "tril"):
                continue
            chain.append(t_i)
            cur = program.nodes[t_i].output
            c_i = _single_consumer(program, cur)
            if c_i is None or not _is_tensorop(program.nodes[c_i], "contiguous"):
                continue
            chain.append(c_i)
            out = program.nodes[c_i].output
            return set(chain), min(chain), out
        return None

    def match(self, program: Program) -> int:
        return 1 if self._region(program) is not None else 0

    def apply(self, program: Program) -> TransformResult:
        region = self._region(program)
        if region is None:
            return TransformResult(program, False, 0, "no canonical KDA chunk_o region")
        drop, insert_at, out = region
        kernel = (_select_prologue_kernel(self.C, self.D) if self.v2 is None
                  else "qk_prologue_v2" if self.v2 else "qk_prologue")
        node = FusedNode(kernel=kernel,
                         inputs=["qF", "kF", "coef_ag", "coef_bg"],
                         outputs=[out],
                         params={"nc": self.nc, "H": self.N, "C": self.C, "D": self.D},
                         subsumes=["chunk_o k_eff mul + Aqk einsum + tril + contiguous"])
        prog = splice(program, drop, insert_at, [node])
        return TransformResult(prog, True, 1, f"per-dim chunk_o glue -> {kernel}")
