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
#  The Program
# --------------------------------------------------------------------------- #
def build_chunked_linear_program(N: int, nc: int, C: int, d_k: int, d_v: int,
                                 work=torch.float16) -> Program:
    """Chunked linear attention over ``N = B*H`` batched heads, ``nc`` chunks.

    Bindings (all batched over ``N``, chunk-major):
        q, k    : [N, nc, C, d_k]      v       : [N, nc, C, d_v]
        P, invP : [N, nc, C, d_k]      gammaInvP : [N, nc, C, d_k]   (= γ/P)
        gamma   : [N, nc, d_k]         (the whole-chunk gate, for the state decay)
    Output:
        o       : [N, nc, C, d_v]
    """
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
    return Program(nodes=nodes,
                   inputs=["q", "k", "v", "P", "invP", "gammaInvP", "gamma"],
                   outputs=["o"])


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
    return dict(q=cast(q), k=cast(k), v=cast(v),
                P=cast(P), invP=cast(invP), gammaInvP=cast(gammaInvP),
                gamma=cast(gamma))


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
