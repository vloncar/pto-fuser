"""The full gated DeltaNet (GDN) forward as one fuser `Program`.

This is the end-to-end counterpart to the per-stage `forwards.deltanet` worked
example: the *complete* gated GDN pipeline (cumsum → kkt → solve_tril → wy_fast →
chunk_h scan → chunk_o), expressed entirely in the IR so the staged / graph-captured
backends run it and it can be benchmarked head-to-head against the megagdn megakernel
(`examples/benchmarks/gdn_mega.py`).

All gating is a set of **host-precomputed coefficient tensors** multiplied into the
matmul operands / scores by ``mul`` Vec glue — the same move `attention/_chunked.py`
makes for the linear-attention decay. The IR only needs einsum cores, ``mul/sub/add/
tril/scale`` glue, and the opaque triangular inverse; the cumulative-sum / ``exp`` gate
arithmetic (which a future glue feature would fold in) is done on the host. The
coefficient decomposition is proven bit-equal to the `RefGDN` fp32 oracle (see
``gdn_reference`` — it reproduces it to fp64 precision).

Simplification vs the full benchmark op: **Hg = H** (no GQA). Megagdn is run with
``key_heads = H`` to match, so the comparison is apples-to-apples.

  kkt       A   = tril( (k·kᵀ) ⊙ exp(gᵢ-gⱼ)·βᵢ , -1)      einsum + mul + tril
  solve     T   = (I + A)⁻¹                                opaque tri_inv
  wy_fast   W,U = T·(k·β·exp(g)), T·(v·β)                  mul + einsum ×2
  chunk_h   per chunk c, state S[D,D]:                     einsum ×2 + mul/sub/add
              vc = U_c - W_c·S ; S = exp(gₗ)·S + kᵀ·(vc·exp(gₗ-g))
  chunk_o   o   = (q·h)·exp(g) + tril(q·kᵀ ⊙ exp(min Δg,0),0)·v_new   einsum ×3 + glue
"""
from __future__ import annotations

import torch

from pto_fuser import (EinsumNode, FusedNode, OpaqueNode, Program, TensorOp,
                       VecGlueNode)


# --------------------------------------------------------------------------- #
#  inputs + fp32 reference (the gate oracle, == RefGDN to fp64)
# --------------------------------------------------------------------------- #
def make_gdn_inputs(B, H, nc, C, D, device, dtype=torch.float16) -> dict:
    """Random GDN operands in [B,H,nc,C,D] / [B,H,nc,C] layout (Hg = H).

    Matches `benchmarks/complex/gdn/utils.generate_random_inputs`: q,k unit-normalized,
    v ~ N(0,1), β ~ U(0,1), gate g = logsigmoid(N(0,1)) (so g ≤ 0, the decay is a
    contraction)."""
    nrm = torch.nn.functional.normalize
    g = dict(device=device, dtype=dtype)
    q = nrm(torch.randn(B, H, nc, C, D, **g), dim=-1)
    k = nrm(torch.randn(B, H, nc, C, D, **g), dim=-1)
    v = torch.randn(B, H, nc, C, D, **g)
    beta = torch.rand(B, H, nc, C, **g)
    g_in = torch.nn.functional.logsigmoid(torch.randn(B, H, nc, C, device=device).float())
    return dict(q=q, k=k, v=v, beta=beta, g_in=g_in)


def _safe_exp(x):
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def gdn_reference(q, k, v, beta, g_in, scale) -> torch.Tensor:
    """fp32 golden, naive per-head — identical to RefGDN.run_full_pipeline (the
    host-coefficient decomposition this Program implements, run densely)."""
    B, H, nc, C, D = q.shape
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    gf = g_in.float()
    g_sum = gf.cumsum(dim=3)                                   # chunk-local cumsum over C
    o = torch.zeros(B, H, nc, C, D, device=q.device)
    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=q.device)
            h_snap = []
            v_new = []
            for c in range(nc):
                gc = g_sum[b, h, c]                            # [C]
                # kkt + solve + wy are chunk-independent; do them inline per chunk.
                A = torch.tril((kf[b, h, c] @ kf[b, h, c].T)
                               * _safe_exp(gc[:, None] - gc[None, :])
                               * bf[b, h, c][:, None], diagonal=-1)
                T = torch.linalg.solve_triangular(
                    torch.eye(C, device=q.device) + A, torch.eye(C, device=q.device),
                    upper=False, unitriangular=True)
                kb = kf[b, h, c] * bf[b, h, c][:, None] * torch.exp(gc)[:, None]
                vb = vf[b, h, c] * bf[b, h, c][:, None]
                W = T @ kb
                U = T @ vb
                h_snap.append(S.clone())
                vc = U - W @ S
                v_new.append(vc)
                gl = gc[-1]
                S = torch.exp(gl) * S + kf[b, h, c].T @ (vc * torch.exp(gl - gc)[:, None])
            for c in range(nc):
                gc = g_sum[b, h, c]
                inter = (qf[b, h, c] @ h_snap[c]) * torch.exp(gc)[:, None]
                qk = qf[b, h, c] @ kf[b, h, c].T
                gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :],
                                               torch.zeros(C, C, device=q.device)))
                causal = (torch.arange(C, device=q.device)[:, None]
                          >= torch.arange(C, device=q.device)[None, :]).float()
                o[b, h, c] = inter + (qk * gate * causal) @ v_new[c]
    return o * scale


# --------------------------------------------------------------------------- #
#  host gate coefficients + Program bindings
# --------------------------------------------------------------------------- #
def prepare_gdn_bindings(q, k, v, beta, g_in, work=torch.float16) -> dict:
    """Flatten operands to the Program's batch layouts and precompute every gate
    coefficient host-side. M = B*H*nc (chunk-independent stages); N = B*H (scan)."""
    B, H, nc, C, D = q.shape
    M = B * H * nc
    g_sum = g_in.float().cumsum(dim=3)                         # [B,H,nc,C]
    gi = g_sum.unsqueeze(-1)                                   # [B,H,nc,C,1]
    gj = g_sum.unsqueeze(-2)                                   # [B,H,nc,1,C]
    bet = beta.float()
    gl = g_sum[..., -1:]                                       # [B,H,nc,1]

    causal = (torch.arange(C, device=q.device)[:, None]
              >= torch.arange(C, device=q.device)[None, :]).float()

    flat = lambda t, *tail: t.reshape(M, *tail).to(work).contiguous()
    coefA = _safe_exp(gi - gj) * bet.unsqueeze(-1)             # [B,H,nc,C,C] kkt score
    coef_kb = (bet * torch.exp(g_sum)).unsqueeze(-1)          # [B,H,nc,C,1] wy w
    coef_vb = bet.unsqueeze(-1)                                # [B,H,nc,C,1] wy u
    coef_qg = torch.exp(g_sum).unsqueeze(-1)                   # [B,H,nc,C,1] chunk_o inter
    coef_o = torch.exp(torch.minimum(gi - gj, torch.zeros_like(gi - gj))) * causal

    # scan coefficients keep the [N,nc,...] shape (sliced per chunk in-graph)
    nflat = lambda t, *tail: t.reshape(B * H, nc, *tail).to(work).contiguous()
    coef_vcs = torch.exp(gl.unsqueeze(-1) - gi)               # [B,H,nc,C,1] exp(gl-g)
    coef_S = torch.exp(gl).unsqueeze(-1)                       # [B,H,nc,1,1] exp(gl)

    return {
        "qF": flat(q, C, D), "kF": flat(k, C, D), "vF": flat(v, C, D),
        "coefA": flat(coefA, C, C), "coef_kb": flat(coef_kb, C, 1),
        "coef_vb": flat(coef_vb, C, 1), "coef_qg": flat(coef_qg, C, 1),
        "coef_o": flat(coef_o, C, C),
        "coef_vcs": nflat(coef_vcs, C, 1), "coef_S": nflat(coef_S, 1, 1),
    }


# --------------------------------------------------------------------------- #
#  the Program
# --------------------------------------------------------------------------- #
def build_gdn_program(B, H, nc, C, D, scale, work=torch.float16,
                      fused_scan=False) -> Program:
    """The full gated GDN forward (output ``o`` [B,H,nc,C,D]). Inputs are the
    flattened operands + gate coefficients from :func:`prepare_gdn_bindings`.

    ``fused_scan`` swaps the unrolled einsum/glue cross-chunk scan (which round-trips
    the recurrent state ``S`` through HBM every chunk) for the ``chunk_h_scan``
    FusedNode, which keeps ``S`` resident on-chip across chunks. GDN's gated recurrence
    ``S = exp(gₗ)·S + kᵀ·((U−W·S)·exp(gₗ−g))`` maps onto the kernel's
    ``S = decay·S + kᵀ·(u−w·S)`` by feeding ``w=W``, ``u=U``, ``k = kF·exp(gₗ−g)`` (the
    per-token factor absorbed into k), ``decay = exp(gₗ)``; the kernel emits the states
    ``h_out`` but not ``v_new``, so ``v_new = U − W·h_out`` is recomputed as one parallel
    batched matmul (no sequential S round-trip). Identical ``o`` either way — this is the
    lowering the fusion-decide gate selects on a measured win."""
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

    inputs = ["qF", "kF", "vF", "coefA", "coef_kb", "coef_vb",
              "coef_qg", "coef_o", "coef_vcs", "coef_S"]

    # -- kkt: A = tril( (k·kᵀ) ⊙ coefA , -1 ) ------------------------------- #
    E("Araw", "nid,njd->nij", "kF", "kF")
    G("Ag", "mul", ["Araw", "coefA"])
    G("A_t", "tril", "Ag", diagonal=-1); T("A", "contiguous", "A_t")

    # -- solve_tril: opaque (I + A)⁻¹ --------------------------------------- #
    Op("T_raw", "tri_inv_rec_unroll", "A"); T("Tm", "cast", "T_raw", dtype=work)

    # -- wy_fast: W = T·(k·coef_kb), U = T·(v·coef_vb) ---------------------- #
    G("kb", "mul", ["kF", "coef_kb"]); G("vb", "mul", ["vF", "coef_vb"])
    E("W_m", "nij,njd->nid", "Tm", "kb"); E("U_m", "nij,njd->nid", "Tm", "vb")
    T("Wb", "reshape", "W_m", shape=[N, nc, C, D])
    T("Ub", "reshape", "U_m", shape=[N, nc, C, D])
    T("kb5", "reshape", "kF", shape=[N, nc, C, D])

    if fused_scan:
        # -- chunk_h scan via FusedNode (S resident on-chip, no per-chunk HBM RT) -- #
        G("k_vcs", "mul", ["kb5", "coef_vcs"])                  # kF·exp(gl-g) [N,nc,C,D]
        for nm, src in (("sw", "Wb"), ("su", "Ub"), ("sk", "k_vcs")):
            T(f"{nm}5", "reshape", src, shape=[B, H, nc, C, D])
            T(nm, "permute", f"{nm}5", dims=(0, 2, 3, 1, 4))   # -> kernel [B,nc,C,H,D]
        T("sdecay", "reshape", "coef_S", shape=[B, H, nc])     # exp(gl) per (b,h,c)
        nodes.append(FusedNode(kernel="chunk_h_scan", inputs=["sw", "su", "sk", "sdecay"],
                               outputs=["h_out_k", "final_k"],
                               params={"B": B, "H": H, "nc": nc},
                               subsumes=["nc per-chunk WS/kv matmul pairs + residual glue"]))
        T("h_bhn", "permute", "h_out_k", dims=(0, 2, 1, 3, 4))  # [B,nc,H,D,D]->[B,H,nc,D,D]
        T("h_flat", "reshape", "h_bhn", shape=[M, D, D])
        E("WS_all", "nid,nde->nie", "W_m", "h_flat")           # parallel W·S (one dispatch)
        G("vn_flat", "sub", ["U_m", "WS_all"])                 # v_new = U - W·S  [M,C,D]
    else:
        # -- chunk_h scan over chunks (resident S in HBM, unrolled) ------------- #
        T("S0", "zeros", "Wb", shape=[N, D, D], dtype=work)
        h_list, vn_list, S = [], [], "S0"
        for c in range(nc):
            h_list.append(S)
            T(f"Wc{c}_s", "slice", "Wb", axis=1, index=c); T(f"Wc{c}", "contiguous", f"Wc{c}_s")
            E(f"WS{c}", "nid,nde->nie", f"Wc{c}", S)
            T(f"Uc{c}", "slice", "Ub", axis=1, index=c)
            G(f"vn{c}", "sub", [f"Uc{c}", f"WS{c}"])                       # vc = U - W·S
            vn_list.append(f"vn{c}")
            T(f"vcs{c}", "slice", "coef_vcs", axis=1, index=c)
            G(f"vn2{c}", "mul", [f"vn{c}", f"vcs{c}"])                     # vc·exp(gl-g)
            T(f"kc{c}_s", "slice", "kb5", axis=1, index=c); T(f"kc{c}", "contiguous", f"kc{c}_s")
            E(f"dS{c}", "nid,nie->nde", f"kc{c}", f"vn2{c}")              # kᵀ·(vc·…)
            T(f"sc{c}", "slice", "coef_S", axis=1, index=c)
            G(f"Sd{c}", "mul", [S, f"sc{c}"])                             # exp(gl)·S
            S = G(f"S{c + 1}", "add", [f"Sd{c}", f"dS{c}"])
        T("h_bh", "stack", h_list, dim=1); T("h_flat", "reshape", "h_bh", shape=[M, D, D])
        T("vn_bh", "stack", vn_list, dim=1); T("vn_flat", "reshape", "vn_bh", shape=[M, C, D])

    # -- chunk_o: o = (q·h)·coef_qg + tril(q·kᵀ ⊙ coef_o)·v_new, scaled ----- #
    E("o_inter_m", "nid,nde->nie", "qF", "h_flat")
    G("o_inter", "mul", ["o_inter_m", "coef_qg"])
    E("Aqk", "nid,njd->nij", "qF", "kF")
    G("Aqk_g", "mul", ["Aqk", "coef_o"]); T("Aqk_c", "contiguous", "Aqk_g")
    E("o_intra", "nij,nje->nie", "Aqk_c", "vn_flat")
    G("o_sum", "add", ["o_inter", "o_intra"], out_dtype=torch.float32)
    G("o_s", "scale", "o_sum", scalar=scale, out_dtype=torch.float32)
    T("o", "reshape", "o_s", shape=[B, H, nc, C, D])

    return Program(nodes=nodes, inputs=inputs, outputs=["o"])
