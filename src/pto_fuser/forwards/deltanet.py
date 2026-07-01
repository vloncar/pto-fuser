"""The chunked DeltaNet forward, expressed as an IR `Program`.

This is the worked example (docs/DESIGN.md §7): the same pipeline as
`prototypes/deltanet_chunk/delta_e2e.py`, but written as a declarative program of
the three compute node types (+ host TensorOps) instead of hardcoded Python, and
run by the generic `StagedExecutor`. Pure DeltaNet (gate g = 0): the per-element
`exp(g)` terms are a proven Vec-scaling class dropped here so the e2e tests exactly
one thing — that the full pipeline COMPOSES across every stage boundary.

    1. kkt        A = tril((βk)@kᵀ, -1)        EinsumNode "nid,njd->nij" + tril glue
    2. solve_tril T = (I + A)^-1               OpaqueNode tri_inv_rec_unroll
    3. recompute  W = T@(βk), U = T@(βv)        EinsumNode "nij,njd->nid" ×2
    4. scan       per chunk c, state S[D,D]:    EinsumNode ×2 + sub/add glue (unrolled)
    5. chunk_o    o = q@h + tril(q@kᵀ)@v_new    EinsumNode ×3 + tril/add/scale glue
"""
from __future__ import annotations

import torch

from ..ir import EinsumNode, OpaqueNode, Program, TensorOp, VecGlueNode


def build_deltanet_program(B: int, H: int, nc: int, C: int, D: int, scale: float,
                           work=torch.float16) -> Program:
    """Build the DeltaNet forward Program. Inputs bound at run time: q, k, v, beta
    (q/k/v: [B,H,nc,C,D]; beta: [B,H,nc,C]). Outputs: A, T, W, U, v_new, h_state, o.

    `work` is the working dtype for einsum/glue results (fp16, matching the
    reference); elementwise residuals accumulate in fp32 inside the glue ops.
    """
    M, BH = B * H * nc, B * H
    nodes: list = []

    def E(out, eq, a, b):
        nodes.append(EinsumNode(eq, [a, b], out, out_dtype=work)); return out

    def G(out, op, ins, out_dtype=work, **p):
        ins = ins if isinstance(ins, list) else [ins]
        nodes.append(VecGlueNode(op, ins, out, params=p, out_dtype=out_dtype)); return out

    def Op(out, key, a):
        nodes.append(OpaqueNode(key, [a], out)); return out

    def T(out, op, ins=None, **p):
        ins = [] if ins is None else (ins if isinstance(ins, list) else [ins])
        nodes.append(TensorOp(op, ins, out, params=p)); return out

    # -- stage 0: β-scale + flatten to [M, C, D] ---------------------------- #
    T("beta_u", "reshape", "beta", shape=[B, H, nc, C, 1])
    G("kb", "mul", ["beta_u", "k"])
    G("vb", "mul", ["beta_u", "v"])
    T("k_h", "cast", "k", dtype=work); T("q_h", "cast", "q", dtype=work)
    for src, dst in [("kb", "kbF"), ("vb", "vbF"), ("k_h", "kF"), ("q_h", "qF")]:
        T(dst + "_r", "reshape", src, shape=[M, C, D]); T(dst, "contiguous", dst + "_r")

    # -- stage 1: kkt -> A = tril(·, -1) ------------------------------------ #
    E("Araw", "nid,njd->nij", "kbF", "kF")
    G("A_t", "tril", "Araw", diagonal=-1); T("A", "contiguous", "A_t")

    # -- stage 2: opaque triangular inverse --------------------------------- #
    Op("T_raw", "tri_inv_rec_unroll", "A"); T("T", "cast", "T_raw", dtype=work)

    # -- stage 3: recompute W, U (two einsums off the same T) --------------- #
    E("W_m", "nij,njd->nid", "T", "kbF"); T("W", "reshape", "W_m", shape=[B, H, nc, C, D])
    E("U_m", "nij,njd->nid", "T", "vbF"); T("U", "reshape", "U_m", shape=[B, H, nc, C, D])
    T("Wb", "reshape", "W", shape=[BH, nc, C, D])
    T("Ub", "reshape", "U", shape=[BH, nc, C, D])
    T("kb5", "reshape", "kF", shape=[BH, nc, C, D])

    # -- stage 4: cross-chunk scan (unrolled over chunks) ------------------- #
    # S carried fp16; residuals accumulate in fp32 then cast back (matches ref).
    T("S0", "zeros", "Wb", shape=[BH, D, D], dtype=work)
    h_list, vn_list, S = [], [], "S0"
    for c in range(nc):
        h_list.append(S)                                    # h_state[c] = S (pre-update)
        T(f"Wc{c}_s", "slice", "Wb", axis=1, index=c); T(f"Wc{c}", "contiguous", f"Wc{c}_s")
        E(f"WS{c}", "nid,nde->nie", f"Wc{c}", S)            # W[c] @ S
        T(f"Uc{c}", "slice", "Ub", axis=1, index=c)
        G(f"vn{c}", "sub", [f"Uc{c}", f"WS{c}"])           # v_new[c] = U[c] - W[c]@S
        vn_list.append(f"vn{c}")
        T(f"kc{c}_s", "slice", "kb5", axis=1, index=c); T(f"kc{c}", "contiguous", f"kc{c}_s")
        E(f"dS{c}", "nid,nie->nde", f"kc{c}", f"vn{c}")    # k[c]ᵀ @ v_new[c]
        S = G(f"S{c + 1}", "add", [S, f"dS{c}"])           # S += dS
    T("h_bh", "stack", h_list, dim=1); T("h_state", "reshape", "h_bh", shape=[B, H, nc, D, D])
    T("vn_bh", "stack", vn_list, dim=1); T("v_new", "reshape", "vn_bh", shape=[B, H, nc, C, D])

    # -- stage 5: chunk_o = q@h_state + tril(q@kᵀ,0) @ v_new, scaled -------- #
    T("h_flat_r", "reshape", "h_bh", shape=[M, D, D]); T("h_flat", "contiguous", "h_flat_r")
    T("vn_flat_r", "reshape", "vn_bh", shape=[M, C, D]); T("vn_flat", "contiguous", "vn_flat_r")
    E("o_inter", "nid,nde->nie", "qF", "h_flat")
    E("Aqk_raw", "nid,njd->nij", "qF", "kF")
    G("Aqk_t", "tril", "Aqk_raw", diagonal=0); T("Aqk", "contiguous", "Aqk_t")
    E("o_intra", "nij,nje->nie", "Aqk", "vn_flat")
    G("o_sum", "add", ["o_inter", "o_intra"], out_dtype=torch.float32)
    G("o_s", "scale", "o_sum", scalar=scale, out_dtype=torch.float32)
    T("o", "reshape", "o_s", shape=[B, H, nc, C, D])

    return Program(nodes=nodes, inputs=["q", "k", "v", "beta"],
                   outputs=["A", "T", "W", "U", "v_new", "h_state", "o"])


def make_inputs(B, H, nc, C, D, device) -> dict:
    """The same random init as delta_e2e.py — small magnitudes keep (I+A)^-1
    well-conditioned and the scan contractive in fp16."""
    nrm = torch.nn.functional.normalize
    g = dict(device=device, dtype=torch.float16)
    k = nrm(torch.randn(B, H, nc, C, D, **g), dim=-1)
    q = nrm(torch.randn(B, H, nc, C, D, **g), dim=-1)
    v = torch.randn(B, H, nc, C, D, **g) * 0.1
    beta = torch.rand(B, H, nc, C, **g) * 0.5
    return dict(q=q, k=k, v=v, beta=beta)


def deltanet_reference(q, k, v, beta, B, H, nc, C, D, scale) -> dict:
    """The identical pipeline in fp32 torch — the gate reference."""
    qf, kf, vf, bf = q.float(), k.float(), v.float(), beta.float()
    kb = bf[..., None] * kf
    vb = bf[..., None] * vf
    flat = lambda t: t.reshape(B * H * nc, C, -1)
    kbF, vbF, kF, qF = flat(kb), flat(vb), flat(kf), flat(qf)
    Araw = torch.einsum("nid,njd->nij", kbF, kF)
    A = torch.tril(Araw, diagonal=-1)
    eye = torch.eye(C, device=q.device).expand(B * H * nc, C, C)
    T = torch.linalg.solve_triangular(eye + A, eye, upper=False, unitriangular=True)
    W = torch.matmul(T, kbF).reshape(B, H, nc, C, D)
    U = torch.matmul(T, vbF).reshape(B, H, nc, C, D)
    kf5 = kf.reshape(B, H, nc, C, D)
    h_state = torch.zeros(B, H, nc, D, D, device=q.device)
    v_new = torch.zeros(B, H, nc, C, D, device=q.device)
    for b in range(B):
        for h in range(H):
            S = torch.zeros(D, D, device=q.device)
            for c in range(nc):
                h_state[b, h, c] = S
                vn = U[b, h, c] - W[b, h, c] @ S
                v_new[b, h, c] = vn
                S = S + kf5[b, h, c].transpose(-1, -2) @ vn
    o_inter = torch.einsum("nid,nde->nie", qF, h_state.reshape(-1, D, D))
    Aqk = torch.tril(torch.einsum("nid,njd->nij", qF, kF), diagonal=0)
    o_intra = torch.einsum("nij,nje->nie", Aqk, v_new.reshape(-1, C, D))
    o = (o_inter + o_intra) * scale
    return dict(A=A, T=T, W=W, U=U, h_state=h_state, v_new=v_new,
                o=o.reshape(B, H, nc, C, D))
