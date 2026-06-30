"""The full Kimi Delta Attention (KDA) forward as one fuser `Program`.

KDA is GDN with a **per-dimension** gate: where GDN carries one scalar decay per
(token, head) and applies it as a post-matmul score factor ``exp(gᵢ-gⱼ)``, KDA carries
a K-vector decay ``g`` of shape ``[…, C, K]`` and bakes it straight into the contraction
*operands* (``k·exp(g)`` against ``k·exp(-g)``). So KDA uses the **same einsum equations
and the same IR vocabulary** as `attention/_gdn_full` — only the host-precomputed gate
coefficients change (they multiply operands, not scores), which is exactly why the
library's direct-read fast paths fire identically for both.

Hg = HV = H (no GQA); K = V = D. Coefficient decomposition proven bit-equal to the
`RefKDA` fp32 oracle (``kda_reference``).

  kkt     A   = tril( (k·exp(g))·(k·exp(-g))ᵀ ⊙ βᵢ , -1)      mul ×2 + einsum + mul + tril
  solve   T   = (I + A)⁻¹                                      opaque tri_inv
  wy      W,U = T·(k·exp(g)·β), T·(v·β)                        mul + einsum ×2
  chunk_h per chunk c, state S[K,V]:                           einsum ×2 + mul/sub/add
            vc = U_c - W_c·S ; S = exp(g_tot)·S + (k·exp(g_tot-g))ᵀ·vc
  chunk_o o   = (q·exp(g))·h + tril((q·exp(g))·(k·exp(-g))ᵀ,0)·v_new   einsum ×3 + glue
"""
from __future__ import annotations

import torch

from pto_fuser import (EinsumNode, FusedNode, OpaqueNode, Program, TensorOp,
                       VecGlueNode)


def make_kda_inputs(B, H, nc, C, D, device, dtype=torch.float16) -> dict:
    """Random KDA operands (Hg = HV = H, K = V = D). Matches
    `benchmarks/complex/kda/utils.generate_random_inputs`: q,k unit-normalized, v ~
    N(0,1), β = sigmoid(N(0,1)), per-dim log-decay g = -U(0,1)·0.05 (small, ≤ 0)."""
    nrm = torch.nn.functional.normalize
    g = dict(device=device, dtype=dtype)
    q = nrm(torch.randn(B, H, nc, C, D, **g), dim=-1)
    k = nrm(torch.randn(B, H, nc, C, D, **g), dim=-1)
    v = torch.randn(B, H, nc, C, D, **g)
    beta = torch.sigmoid(torch.randn(B, H, nc, C, **g))
    g_in = -torch.rand(B, H, nc, C, D, device=device).float() * 0.05      # per-dim
    return dict(q=q, k=k, v=v, beta=beta, g_in=g_in)


def kda_reference(q, k, v, beta, g_in, scale) -> torch.Tensor:
    """fp32 golden, naive per-head — identical to RefKDA.run_full_pipeline."""
    B, H, nc, C, D = q.shape
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    g_sum = g_in.float().cumsum(dim=3)                        # [B,H,nc,C,D] per-dim cumsum
    eye = torch.eye(C, device=q.device)
    causal = (torch.arange(C, device=q.device)[:, None]
              >= torch.arange(C, device=q.device)[None, :]).float()
    o = torch.zeros(B, H, nc, C, D, device=q.device)
    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=q.device)            # [K, V]
            snaps, vcorrs = [], []
            for c in range(nc):
                gc = g_sum[b, h, c]                            # [C, K]
                a_op = kf[b, h, c] * torch.exp(gc)
                b_op = kf[b, h, c] * torch.exp(-gc)
                A = torch.tril((a_op @ b_op.T) * bf[b, h, c][:, None], diagonal=-1)
                T = torch.linalg.solve_triangular(eye + A, eye, upper=False,
                                                  unitriangular=True)
                W = T @ (kf[b, h, c] * torch.exp(gc) * bf[b, h, c][:, None])
                U = T @ (vf[b, h, c] * bf[b, h, c][:, None])
                snaps.append(S.clone())
                vc = U - W @ S
                vcorrs.append(vc)
                g_tot = gc[-1]                                 # [K]
                k_rest = kf[b, h, c] * torch.exp(g_tot[None, :] - gc)
                S = torch.exp(g_tot)[:, None] * S + k_rest.T @ vc
            for c in range(nc):
                gc = g_sum[b, h, c]
                q_eff = qf[b, h, c] * torch.exp(gc)
                k_eff = kf[b, h, c] * torch.exp(-gc)
                Aqk = (q_eff @ k_eff.T) * causal
                o[b, h, c] = q_eff @ snaps[c] + Aqk @ vcorrs[c]
    return o * scale


def prepare_kda_bindings(q, k, v, beta, g_in, work=torch.float16) -> dict:
    """Flatten operands + precompute per-dim gate coefficients. M = B*H*nc, N = B*H."""
    B, H, nc, C, D = q.shape
    M = B * H * nc
    g_sum = g_in.float().cumsum(dim=3)                        # [B,H,nc,C,D]
    bet = beta.float().unsqueeze(-1)                          # [B,H,nc,C,1]
    g_tot = g_sum[:, :, :, -1:, :]                            # [B,H,nc,1,D]

    flat = lambda t, *tail: t.reshape(M, *tail).to(work).contiguous()
    nflat = lambda t, *tail: t.reshape(B * H, nc, *tail).to(work).contiguous()

    coef_ag = torch.exp(g_sum)                                # exp(g)  : q_eff / a_op
    coef_bg = torch.exp(-g_sum)                               # exp(-g) : k_eff / b_op
    coef_wg = torch.exp(g_sum) * bet                          # exp(g)·β: wy w
    coef_krest = torch.exp(g_tot - g_sum)                     # exp(g_tot-g): scan k
    coef_S = torch.exp(g_tot).reshape(B, H, nc, D, 1)         # exp(g_tot): scan S rows

    return {
        "qF": flat(q, C, D), "kF": flat(k, C, D), "vF": flat(v, C, D),
        "coef_ag": flat(coef_ag, C, D), "coef_bg": flat(coef_bg, C, D),
        "coef_beta": flat(bet.expand(B, H, nc, C, 1), C, 1),
        "coef_wg": flat(coef_wg, C, D),
        "coef_ub": flat(bet.expand(B, H, nc, C, 1), C, 1),
        "coef_krest": nflat(coef_krest, C, D), "coef_S": nflat(coef_S, D, 1),
    }


def build_kda_program(B, H, nc, C, D, scale, work=torch.float16,
                      fused_scan=False) -> Program:
    """The full KDA forward (output ``o`` [B,H,nc,C,D]). Same equations as the GDN
    forward; the gate enters via operand muls instead of score muls.

    ``fused_scan`` swaps the unrolled einsum/glue cross-chunk scan for the
    ``chunk_h_scan`` FusedNode (resident state, no per-chunk HBM round-trip), exactly
    as the GDN forward does. The one KDA difference is the cross-chunk decay: GDN's is
    a scalar ``exp(gₗ)`` per (b,h,c); KDA's is the **per-dimension** vector
    ``exp(g_tot)`` (shape ``[D]``), decaying each K-row of ``S`` independently — so the
    FusedNode is built with ``perdim_decay=True`` (selecting the kernel's
    ``SCAN_PERDIM_DECAY`` variant, which row-broadcasts the decay vector across ``S``).
    The within-chunk per-dim factor ``exp(g_tot−g)`` is absorbed into ``k`` (the
    ``coef_krest`` mul), and ``v_new = U − W·S`` is recovered as one parallel batched
    matmul. Identical ``o`` either way — the fusion-decide gate selects it on a win."""
    M, N = B * H * nc, B * H
    nodes: list = []

    def E(out, eq, a, b):
        nodes.append(EinsumNode(eq, [a, b], out, out_dtype=work)); return out

    def G(out, op, ins, out_dtype=work, **p):
        nodes.append(VecGlueNode(op, ins if isinstance(ins, list) else [ins], out,
                                 params=p, out_dtype=out_dtype)); return out

    def Op(out, key, a):
        nodes.append(OpaqueNode(key, [a], out)); return out

    def T(out, op, ins=None, **p):
        ins = [] if ins is None else (ins if isinstance(ins, list) else [ins])
        nodes.append(TensorOp(op, ins, out, params=p)); return out

    inputs = ["qF", "kF", "vF", "coef_ag", "coef_bg", "coef_beta", "coef_wg",
              "coef_ub", "coef_krest", "coef_S"]

    # -- kkt: A = tril( (k·exp(g))·(k·exp(-g))ᵀ ⊙ β , -1 ) ------------------ #
    G("a_op", "mul", ["kF", "coef_ag"]); G("b_op", "mul", ["kF", "coef_bg"])
    E("Araw", "nid,njd->nij", "a_op", "b_op")
    G("Ag", "mul", ["Araw", "coef_beta"])
    G("A_t", "tril", "Ag", diagonal=-1); T("A", "contiguous", "A_t")

    # -- solve_tril: opaque (I + A)⁻¹ --------------------------------------- #
    Op("T_raw", "tri_inv_rec_unroll", "A"); T("Tm", "cast", "T_raw", dtype=work)

    # -- wy: W = T·(k·exp(g)·β), U = T·(v·β) -------------------------------- #
    G("wk", "mul", ["kF", "coef_wg"]); G("vb", "mul", ["vF", "coef_ub"])
    E("W_m", "nij,njd->nid", "Tm", "wk"); E("U_m", "nij,njd->nid", "Tm", "vb")
    T("Wb", "reshape", "W_m", shape=[N, nc, C, D])
    T("Ub", "reshape", "U_m", shape=[N, nc, C, D])
    T("kb5", "reshape", "kF", shape=[N, nc, C, D])

    if fused_scan:
        # -- chunk_h scan via FusedNode (S resident on-chip, per-dim decay) ----- #
        G("k_krest", "mul", ["kb5", "coef_krest"])             # kF·exp(g_tot-g) [N,nc,C,D]
        for nm, src in (("sw", "Wb"), ("su", "Ub"), ("sk", "k_krest")):
            T(f"{nm}5", "reshape", src, shape=[B, H, nc, C, D])
            T(nm, "permute", f"{nm}5", dims=(0, 2, 3, 1, 4))   # -> kernel [B,nc,C,H,D]
        T("sdecay", "reshape", "coef_S", shape=[B, H, nc, D])  # exp(g_tot) per (b,h,c,dim)
        nodes.append(FusedNode(kernel="chunk_h_scan", inputs=["sw", "su", "sk", "sdecay"],
                               outputs=["h_out_k", "final_k"],
                               params={"B": B, "H": H, "nc": nc, "perdim_decay": True},
                               subsumes=["nc per-chunk WS/kv matmul pairs + residual glue"]))
        T("h_bhn", "permute", "h_out_k", dims=(0, 2, 1, 3, 4))  # [B,nc,H,D,D]->[B,H,nc,D,D]
        T("h_flat", "reshape", "h_bhn", shape=[M, D, D])
        E("WS_all", "nid,nde->nie", "W_m", "h_flat")           # parallel W·S (one dispatch)
        G("vn_flat", "sub", ["U_m", "WS_all"])                 # v_new = U - W·S  [M,C,D]
    else:
        # -- chunk_h scan (resident S in HBM, unrolled) ------------------------- #
        T("S0", "zeros", "Wb", shape=[N, D, D], dtype=work)
        h_list, vn_list, S = [], [], "S0"
        for c in range(nc):
            h_list.append(S)
            T(f"Wc{c}_s", "slice", "Wb", axis=1, index=c); T(f"Wc{c}", "contiguous", f"Wc{c}_s")
            E(f"WS{c}", "nid,nde->nie", f"Wc{c}", S)
            T(f"Uc{c}", "slice", "Ub", axis=1, index=c)
            G(f"vn{c}", "sub", [f"Uc{c}", f"WS{c}"])                       # vc = U - W·S
            vn_list.append(f"vn{c}")
            T(f"kc{c}_s", "slice", "kb5", axis=1, index=c); T(f"kc{c}", "contiguous", f"kc{c}_s")
            T(f"krc{c}", "slice", "coef_krest", axis=1, index=c)
            G(f"krest{c}", "mul", [f"kc{c}", f"krc{c}"])                   # k·exp(g_tot-g)
            E(f"dS{c}", "nid,nie->nde", f"krest{c}", f"vn{c}")            # kᵀ·vc
            T(f"sc{c}", "slice", "coef_S", axis=1, index=c)
            G(f"Sd{c}", "mul", [S, f"sc{c}"])                             # exp(g_tot)·S
            S = G(f"S{c + 1}", "add", [f"Sd{c}", f"dS{c}"])
        T("h_bh", "stack", h_list, dim=1); T("h_flat", "reshape", "h_bh", shape=[M, D, D])
        T("vn_bh", "stack", vn_list, dim=1); T("vn_flat", "reshape", "vn_bh", shape=[M, C, D])

    # -- chunk_o: o = (q·exp(g))·h + tril((q·exp(g))·(k·exp(-g))ᵀ,0)·v_new --- #
    G("q_eff", "mul", ["qF", "coef_ag"]); G("k_eff", "mul", ["kF", "coef_bg"])
    E("o_inter", "nid,nde->nie", "q_eff", "h_flat")
    E("Aqk", "nid,njd->nij", "q_eff", "k_eff")
    G("Aqk_t", "tril", "Aqk", diagonal=0); T("Aqk_c", "contiguous", "Aqk_t")
    E("o_intra", "nij,nje->nie", "Aqk_c", "vn_flat")
    G("o_sum", "add", ["o_inter", "o_intra"], out_dtype=torch.float32)
    G("o_s", "scale", "o_sum", scalar=scale, out_dtype=torch.float32)
    T("o", "reshape", "o_s", shape=[B, H, nc, C, D])

    return Program(nodes=nodes, inputs=inputs, outputs=["o"])
