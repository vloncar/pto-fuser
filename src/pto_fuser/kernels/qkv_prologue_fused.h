#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Per-dim flash — the PER-CHANNEL-gate counterpart of qkv_fused.h (B3 continuation).
//
// KDA/GLA chunk_o is q_eff=q⊙coef_ag, k_eff=k⊙coef_bg → A=tril(q_eff·k_effᵀ) →
// o=A·v. The scalar flash (qkv_fused.h) applies the gate as a Vec EPILOGUE after the
// score matmul; here the per-channel decay rides on the OPERANDS (a Vec PROLOGUE)
// before it — no scalar coeff reproduces it. So the pipeline gains a leading Vec
// prescale stage: Vec(prescale) → Cube(S) → Vec(tril) → Cube(o), and the [M,C,C]
// masked score still never lands in HBM.
//
// Reuses `prologue_fused::prescale_one` (q⊙coef_ag, k⊙coef_bg) and
// `prologue_fused::mask_one` (tril, no gate coeff) verbatim, plus the SAME two configs
// as qkv_fused.h (QK_CFG native-NT for the prescale + score, AV_CFG native-NN for
// A·v). Only-new device code = the 4-stage choreography.
//
// ── V1 (this file): four-pass — the bit-exact ORACLE for V2 ───────────────────
//   Vec prescale ALL -> qd_s/kinv_s; Cube S=q̃@k̂ᵀ ALL -> ws_qk; Vec tril ALL -> ws_A;
//   Cube o=A@v ALL -> o. Correct; materializes the scratch (no bandwidth win — that
//   is the interleaved V2).
//
// Operands (host, native [M,C,D]): q,k,v [M,C,D] half; coef_ag/coef_bg [M,C,D] half
// (the per-dim exp(±g) decay); mask [C,C] float causal (i>=j). Output o [M,C,DV] f32.
// ─────────────────────────────────────────────────────────────────────────────
#include "prologue_fused.h"

namespace qkv_prologue_fused {

using namespace pto;

// V1: Vec prescale -> Cube S -> Vec tril -> Cube o, three SyncAlls.
//   QK_CFG: native-NT [M,C,D] config (drives BOTH prescale_one and the S matmul).
//   AV_CFG: native-NN [M,C,C]@[M,C,DV] output config.
template <typename QK_CFG, typename AV_CFG>
__global__ AICORE void qkv_prologue_flash_v1(__gm__ const half* q, __gm__ const half* coef_ag,
                                             __gm__ const half* k, __gm__ const half* coef_bg,
                                             __gm__ const float* mask, __gm__ half* qd_s,
                                             __gm__ half* kinv_s, __gm__ float* ws_qk,
                                             __gm__ half* ws_A, __gm__ const half* v,
                                             __gm__ float* o, uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = QK_CFG::n_free0;
    constexpr unsigned I = QK_CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned iters = (I + bnum - 1) / bnum;

    // Stage 1: q̃ = q⊙coef_ag -> qd_s, k̂ = k⊙coef_bg -> kinv_s.
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            if (pid < I)
                prologue_fused::prescale_one<QK_CFG>(q, coef_ag, k, coef_bg,
                                                     qd_s, kinv_s, pid, vid);
        }
    }
    pto_einsum::SyncAll<false>();
    // Stage 2: S = q̃ @ k̂ᵀ (NT direct-read of the prescaled scratch).
    if constexpr (DAV_CUBE) {
        pto_einsum::batched_matmul_inline<half, QK_CFG, false, false, true>(
            qd_s, kinv_s, ws_qk, get_block_idx(), get_block_num());
    }
    pto_einsum::SyncAll<false>();
    // Stage 3: A = tril(S) -> ws_A half.
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            if (pid < I)
                prologue_fused::mask_one<QK_CFG>(ws_qk + (int64_t)pid * C * C, mask,
                                                 ws_A, pid, vid);
        }
    }
    pto_einsum::SyncAll<false>();
    // Stage 4: o = A @ v (NN contiguous).
    if constexpr (DAV_CUBE) {
        pto_einsum::batched_matmul_inline<half, AV_CFG, false, false, false>(
            ws_A, v, o, get_block_idx(), get_block_num());
    }
}

// ── V2: double-buffered 4-stage interleave — ops, S, A never leave L2 ─────────
// FFTS flag ids (base + slot), disjoint from SyncAll (11-14); this kernel runs no
// other protocol in its launch. Three handoffs: Vec→Cube ops-ready, Cube→Vec S-ready,
// Vec→Cube A-ready. Two slots each so tile w+1's prescale/score overlap tile w's mask/
// output; the WAR reuses are ordered by each unit's own sequential stream (as in the
// scalar flash), so no FREE flags are needed.
constexpr uint16_t QKVP_OP_READY = 16;   // Vec -> Cube: prescaled q̃/k̂ ready (16,17)
constexpr uint16_t QKVP_S_READY  = 18;   // Cube -> Vec: score S ready      (18,19)
constexpr uint16_t QKVP_A_READY  = 20;   // Vec -> Cube: masked A ready      (20,21)

// COUNT: the batched config (supplies I,C,D). QK_TILE/AV_TILE: single-batch NT/NN
// configs read from the per-core ring slots (base offset 0).
template <typename COUNT, typename QK_TILE, typename AV_TILE>
__global__ AICORE void qkv_prologue_flash_v2(__gm__ const half* q, __gm__ const half* coef_ag,
                                             __gm__ const half* k, __gm__ const half* coef_bg,
                                             __gm__ const float* mask, __gm__ half* op_ring,
                                             __gm__ float* s_ring, __gm__ half* a_ring,
                                             __gm__ const half* v, __gm__ float* o,
                                             uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = COUNT::n_free0;
    constexpr unsigned D = COUNT::n_contract;
    constexpr unsigned DV = AV_TILE::n_free1;
    constexpr unsigned I = COUNT::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned iters = (I + bnum - 1) / bnum;
    // Per-core: 2 op slots (qd|kinv, C*D halfs each), 2 S slots (C*C f32), 2 A slots (C*C f16).
    __gm__ half*  OP = op_ring + (int64_t)cid * 2 * (2 * C * D);
    __gm__ float* SR = s_ring + (int64_t)cid * 2 * C * C;
    __gm__ half*  AR = a_ring + (int64_t)cid * 2 * C * C;
    // op slot s: qd at OP + s*2*C*D, kinv at + C*D.
#define QKVP_OPA(s) (OP + (int64_t)(s) * 2 * C * D)
#define QKVP_OPB(s) (OP + (int64_t)(s) * 2 * C * D + (int64_t)C * D)

    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        // prologue: prescale tile 0 so the Cube's first score can start.
        if (cid < I)
            prologue_fused::prescale_to_slot<QK_TILE>(q, coef_ag, k, coef_bg,
                                                      QKVP_OPA(0), QKVP_OPB(0), cid, vid);
        pipe_barrier(PIPE_ALL);
        ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, QKVP_OP_READY + 0));
        for (unsigned w = 0; w < iters; ++w) {
            unsigned slot = w & 1u;
            if (w + 1 < iters) {                       // prescale tile w+1 (overlaps Cube S(w))
                unsigned nslot = (w + 1) & 1u;
                unsigned npid = (w + 1) * bnum + cid;
                if (npid < I)
                    prologue_fused::prescale_to_slot<QK_TILE>(q, coef_ag, k, coef_bg,
                                                              QKVP_OPA(nslot), QKVP_OPB(nslot), npid, vid);
                pipe_barrier(PIPE_ALL);
                ffts_cross_core_sync(PIPE_MTE3,
                                     pto_einsum::GetffstMsg(0x2, QKVP_OP_READY + nslot));
            }
            unsigned pid = w * bnum + cid;
            wait_flag_dev(QKVP_S_READY + slot);
            pipe_barrier(PIPE_ALL);
            if (pid < I)                               // A = tril(S) -> A slot (pointer trick)
                prologue_fused::mask_one<QK_TILE>(SR + (int64_t)slot * C * C, mask,
                                                  (AR + (int64_t)slot * C * C) - (int64_t)pid * C * C,
                                                  pid, vid);
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, QKVP_A_READY + slot));
        }
    }
    if constexpr (DAV_CUBE) {
        // prologue: S(0) = q̃(0) @ k̂(0)ᵀ from op slot 0.
        wait_flag_dev(QKVP_OP_READY + 0);
        pipe_barrier(PIPE_ALL);
        if (cid < I)
            pto_einsum::matmul_one_tile_deep<half, QK_TILE, false, true, true>(
                QKVP_OPA(0), QKVP_OPB(0), nullptr, 0, SR);
        ffts_cross_core_sync(PIPE_FIX, pto_einsum::GetffstMsg(0x2, QKVP_S_READY + 0));
        for (unsigned w = 0; w < iters; ++w) {
            unsigned slot = w & 1u;
            if (w + 1 < iters) {                       // score tile w+1 (overlaps Vec mask(w))
                unsigned nslot = (w + 1) & 1u;
                unsigned npid = (w + 1) * bnum + cid;
                wait_flag_dev(QKVP_OP_READY + nslot);
                pipe_barrier(PIPE_ALL);
                if (npid < I)
                    pto_einsum::matmul_one_tile_deep<half, QK_TILE, false, true, true>(
                        QKVP_OPA(nslot), QKVP_OPB(nslot), nullptr, 0, SR + (int64_t)nslot * C * C);
                ffts_cross_core_sync(PIPE_FIX,
                                     pto_einsum::GetffstMsg(0x2, QKVP_S_READY + nslot));
            }
            unsigned pid = w * bnum + cid;
            wait_flag_dev(QKVP_A_READY + slot);
            pipe_barrier(PIPE_ALL);
            if (pid < I)                               // o(w) = A(w) @ v[pid]
                pto_einsum::matmul_one_tile_deep<half, AV_TILE, false, false, true>(
                    AR + (int64_t)slot * C * C, v + (int64_t)pid * C * DV, nullptr, 0,
                    o + (int64_t)pid * C * DV);
        }
    }
#undef QKVP_OPA
#undef QKVP_OPB
}

} // namespace qkv_prologue_fused
