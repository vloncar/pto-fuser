"""Chunked linear attention — the shared core behind the linear-recurrence family.

Vanilla linear attention, RetNet, GLA and Mamba-2 (SSD) are *the same chunked
recurrence* with different decay. This module implements that recurrence once as a
pto-fuser ``Program`` and lets each example supply its own per-token gate; the
variant files are then a few lines each.

The math (per head, state ``S`` of shape ``[d_k, d_v]``, chunk size ``C``)
---------------------------------------------------------------------------
Token-recurrent definition (this is the fp32 reference):

    S_t = diag(g_t) · S_{t-1} + k_tᵀ v_t          # g_t ∈ (0,1]^{d_k}, the gate
    o_t = q_t · S_t                                # causal, current token included

Chunk-parallel form (this is the Program). Inside a chunk, let ``P_i`` be the
inclusive cumulative gate ``∏_{j≤i} g_j`` (a host-side scan — the piece a cumsum
glue feature would absorb), and ``γ`` the whole-chunk gate ``P_{C-1}``. Define the
decayed query / inverse-decayed key / state-update key:

    q̃_i = q_i ⊙ P_i           k̂_j = k_j ⊙ (1/P_j)        k̄_j = k_j ⊙ (γ/P_j)

Then, with incoming state ``S_in``:

    O_intra = tril(q̃ k̂ᵀ, 0) · V            # within-chunk attention with decay
    O_inter = q̃ · S_in                       # contribution of the carried state
    O       = O_inter + O_intra
    S_out   = diag(γ) · S_in + k̄ᵀ · V        # state handed to the next chunk

Specialisations (just the gate ``g_t``):

    vanilla LA : g_t = 1                        (no decay)
    RetNet     : g_t = γ_head            (scalar constant per head, broadcast over d_k)
    Mamba-2    : g_t = a_t               (scalar per token per head, data-dependent)
    GLA        : g_t = σ(·)_t            (per-channel per token, data-dependent)

The decay tensors (``P``, ``1/P``, ``γ/P``, ``γ``) are precomputed on the host in
:func:`prepare_inputs`; the Program itself is pure einsum cores + Vec ``mul``/``add``
+ a triangular mask + the cross-chunk scan, so it runs on the unmodified executor and
captures/fuses like any other forward.

> Numerical note: ``1/P`` grows as gates shrink, so keep chunks modest and gates
> near 1 (the examples use ``C=16`` and ``g ≥ 0.95``). Production GLA adds a secondary
> intra-chunk split to bound this; that is an orthogonal optimization left out here.
"""
from __future__ import annotations

import torch

from pto_fuser import EinsumNode, Program, TensorOp, VecGlueNode


# --------------------------------------------------------------------------- #
#  Per-dim prologue kernel selection (shape gate)
# --------------------------------------------------------------------------- #
def select_prologue_kernel(C: int, d_k: int) -> str:
    """Pick the per-dim prologue lowering by shape.

    Both kernels compute the same ``tril((q⊙P)(k⊙invP)ᵀ)``; they differ in where the
    scaled operands live. ``qk_prologue_v2`` keeps them in an L2-resident ring (no full
    ``[M,C,D]`` qd/kinv scratch) and wins where the prescale is bandwidth-heavy — large
    score/head dims (measured 2.8–5.8× over staged at ``C·d_k ≥ 4096``). At tiny tiles
    the per-tile single-tile matmul granularity makes V2 lose to V1's batched two-pass,
    so V1 is the default there. (A double-buffered V2 was measured to add only ~1.1× in
    the win regime and can't flip the tiny-tile loss — see the fusion notes — so it is
    not a third option.) ``fusion.decide`` can override this by measurement."""
    return "qk_prologue_v2" if C * d_k >= 4096 else "qk_prologue"


# --------------------------------------------------------------------------- #
#  The Program
# --------------------------------------------------------------------------- #
def build_chunked_linear_program(N: int, nc: int, C: int, d_k: int, d_v: int,
                                 work=torch.float16) -> Program:
    """Chunked linear attention over ``N = B*H`` batched heads, ``nc`` chunks, as the
    **canonical** (all-staged) `Program`.

    Bindings (all batched over ``N``, chunk-major):
        q, k    : [N, nc, C, d_k]      v       : [N, nc, C, d_v]
        P, invP : [N, nc, C, d_k]      gammaInvP : [N, nc, C, d_k]   (= γ/P)
        gamma   : [N, nc, d_k]         (the whole-chunk gate, for the state decay)
        g_intra : [N·nc, C]            (scalar log-cumgate) + beta_intra ones — declared
                                       so the batch-chunk-intra-score transform validates
    Output:
        o       : [N, nc, C, d_v]

    The intra-chunk score ``A_c = tril(q̃_c k̂_cᵀ, 0)`` is unrolled per chunk here; the
    ``batch-chunk-intra-score`` transform (``pto_fuser.transforms.chunked``) collapses all
    ``nc`` into one batched proven kernel (``gated_qk_native_v2`` for a scalar gate, or
    ``qk_prologue`` for a per-channel gate — the gate kind is a compile option, not a
    build flag)."""
    nodes: list = []

    def E(out, eq, a, b):
        nodes.append(EinsumNode(eq, [a, b], out, out_dtype=work)); return out

    def G(out, op, ins, out_dtype=work, **p):
        nodes.append(VecGlueNode(op, ins, out, params=p, out_dtype=out_dtype)); return out

    def T(out, op, ins, **p):
        ins = ins if isinstance(ins, list) else [ins]
        nodes.append(TensorOp(op, ins, out, params=p)); return out

    T("S0", "zeros", "q", shape=(N, d_k, d_v), dtype=work)
    outs, S = [], "S0"
    for c in range(nc):
        # slice this chunk's operands and decay tensors
        for src in ("q", "k", "v", "P", "invP", "gammaInvP"):
            T(f"{src}{c}", "slice", src, axis=1, index=c)
        T(f"g{c}", "slice", "gamma", axis=1, index=c)             # [N, d_k]

        # decay application (Vec): q̃ = q⊙P, k̂ = k⊙(1/P), k̄ = k⊙(γ/P)
        G(f"qd{c}", "mul", [f"q{c}", f"P{c}"])
        G(f"kinv{c}", "mul", [f"k{c}", f"invP{c}"])
        G(f"kout{c}", "mul", [f"k{c}", f"gammaInvP{c}"])

        # intra-chunk attention with decay:  tril(q̃ k̂ᵀ, 0) · V
        E(f"A{c}", "nid,njd->nij", f"qd{c}", f"kinv{c}")
        G(f"Am{c}", "tril", [f"A{c}"], diagonal=0)
        E(f"o_intra{c}", "nij,nje->nie", f"Am{c}", f"v{c}")
        # inter-chunk:  q̃ · S_in
        E(f"o_inter{c}", "nid,nde->nie", f"qd{c}", S)
        outs.append(G(f"o{c}", "add", [f"o_inter{c}", f"o_intra{c}"]))

        # state update:  S' = diag(γ)·S + k̄ᵀ·V
        E(f"dS{c}", "nid,nie->nde", f"kout{c}", f"v{c}")
        T(f"gcol{c}", "reshape", f"g{c}", shape=(N, d_k, 1))
        G(f"Sdec{c}", "mul", [S, f"gcol{c}"])
        S = G(f"S{c+1}", "add", [f"Sdec{c}", f"dS{c}"], out_dtype=work)

    T("o", "stack", outs, dim=1)                                  # [N, nc, C, d_v]
    # g_intra/beta_intra are consumed only after the scalar batch-chunk-intra-score
    # transform fires; declared here (like GDN's g_native) so that rewrite validates.
    inputs = ["q", "k", "v", "P", "invP", "gammaInvP", "gamma", "g_intra", "beta_intra"]
    return Program(nodes=nodes, inputs=inputs, outputs=["o"])


# --------------------------------------------------------------------------- #
#  Fusion decision for the intra-score lever
# --------------------------------------------------------------------------- #
def decide_fused_intra(N, nc, C, d_k, d_v, gates, dev, *, per_dim_gate, iters=20):
    """Gate + measure the fused intra-score lever against the staged per-chunk
    ``einsum + tril``, via :func:`pto_fuser.decide` (needs an NPU).

    Runs the two lowerings of the whole chunked forward on identical inputs — staged vs
    ``fused_intra`` — captures each, and returns the :class:`FusionDecision`. The scalar
    family (``per_dim_gate=False``) fuses the gated EPILOGUE (``gated_qk_native_v2``);
    GLA/KDA (``per_dim_gate=True``) fuse the operand PROLOGUE (``qk_prologue[_v2]``,
    shape-selected). The fuser keeps the fused kernel only on a gated-green + deterministic
    + faster win — otherwise the staged-captured lowering stands (the lever ordering)."""
    from pto_fuser import GraphReplayExecutor, decide
    from pto_fuser.transforms import BatchChunkIntraScore
    q, k, v = make_qkv(N, nc, C, d_k, d_v, dev)
    binds = prepare_inputs(q, k, v, gates)
    canon = build_chunked_linear_program(N, nc, C, d_k, d_v)
    fused_prog = BatchChunkIntraScore(N, nc, C, d_k, d_v,
                                      per_dim_gate=per_dim_gate).apply(canon).program
    staged = GraphReplayExecutor().capture(canon, binds)
    fused = GraphReplayExecutor().capture(fused_prog, binds)
    kern = select_prologue_kernel(C, d_k) if per_dim_gate else "gated_qk_native_v2"
    return decide("chunk_intra", kern,
                  lambda: staged.replay(binds, clone=False),
                  lambda: fused.replay(binds, clone=False), iters=iters)


# --------------------------------------------------------------------------- #
#  Host input prep + reference
# --------------------------------------------------------------------------- #
def prepare_inputs(q, k, v, gates, *, work=torch.float16) -> dict:
    """Turn per-token ``[N, nc, C, d_k]`` gates into the chunk decay tensors the
    Program binds. ``q/k/v`` are ``[N, nc, C, d_*]``. The inclusive cumulative gate
    ``P`` is the host-side scan (a cumsum a glue feature could absorb)."""
    gf = gates.float().clamp_min(1e-6)
    P = torch.cumprod(gf, dim=2)                          # inclusive, within chunk
    gamma = P[:, :, -1, :]                                # whole-chunk gate [N,nc,d_k]
    invP = 1.0 / P
    gammaInvP = gamma.unsqueeze(2) * invP
    cast = lambda t: t.to(work)
    # Scalar log-cumgate for the fused gated_qk epilogue (build_chunked_linear_program
    # fused_intra=True). g = log P, reduced over d_k (exact for a scalar gate — all
    # channels equal; the geometric mean for a per-channel gate, which must NOT fuse).
    N, nc = P.shape[0], P.shape[1]
    g_intra = torch.log(P).mean(dim=-1).reshape(N * nc, P.shape[2]).float().contiguous()
    beta_intra = torch.ones(N * nc, P.shape[2], device=P.device, dtype=work)
    return dict(q=cast(q), k=cast(k), v=cast(v),
                P=cast(P), invP=cast(invP), gammaInvP=cast(gammaInvP),
                gamma=cast(gamma), g_intra=g_intra, beta_intra=beta_intra)


def chunked_linear_torch(q, k, v, gates, B, H, nc, C) -> dict:
    """Eager-torch mirror of :func:`build_chunked_linear_program` — the same chunked
    formula in plain torch (runs on CPU). Used as a readable spec and to test the
    chunk decomposition off-NPU, where it must match :func:`chunked_linear_reference`.
    """
    N = B * H
    d_v = v.shape[-1]
    qf, kf, vf, gf = (t.float() for t in (q, k, v, gates))
    P = torch.cumprod(gf.clamp_min(1e-6), dim=2)
    gamma = P[:, :, -1, :]
    invP = 1.0 / P
    gammaInvP = gamma.unsqueeze(2) * invP
    o = torch.zeros(N, nc, C, d_v, device=q.device)
    S = torch.zeros(N, q.shape[-1], d_v, device=q.device)
    for c in range(nc):
        qd = qf[:, c] * P[:, c]
        kinv = kf[:, c] * invP[:, c]
        kout = kf[:, c] * gammaInvP[:, c]
        A = torch.tril(torch.einsum("nid,njd->nij", qd, kinv), diagonal=0)
        o[:, c] = torch.einsum("nij,nje->nie", A, vf[:, c]) + torch.einsum("nid,nde->nie", qd, S)
        S = S * gamma[:, c].unsqueeze(-1) + torch.einsum("nid,nie->nde", kout, vf[:, c])
    return {"o": o}


def chunked_linear_reference(q, k, v, gates, B, H, nc, C) -> dict:
    """Token-recurrent fp32 reference: ``S_t = diag(g_t) S_{t-1} + k_tᵀ v_t``,
    ``o_t = q_t S_t``. The chunked Program must reproduce this bit-faithfully (the
    gate proves the chunk decomposition is correct)."""
    N = B * H
    d_k, d_v = q.shape[-1], v.shape[-1]
    qf, kf, vf, gf = (t.float().reshape(N, nc * C, -1) for t in (q, k, v, gates))
    o = torch.zeros(N, nc * C, d_v, device=q.device)
    for n in range(N):
        S = torch.zeros(d_k, d_v, device=q.device)
        for t in range(nc * C):
            S = gf[n, t][:, None] * S + kf[n, t][:, None] * vf[n, t][None, :]
            o[n, t] = qf[n, t] @ S
    return {"o": o.reshape(N, nc, C, d_v)}


def make_qkv(N, nc, C, d_k, d_v, device, seed=0):
    """Small, well-scaled q/k/v so the fp16 chunked form stays close to fp32."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: (torch.randn(*s, generator=g) * 0.2).to(device)
    q = torch.nn.functional.normalize(mk(N, nc, C, d_k), dim=-1)
    k = torch.nn.functional.normalize(mk(N, nc, C, d_k), dim=-1)
    v = mk(N, nc, C, d_v)
    return q, k, v
