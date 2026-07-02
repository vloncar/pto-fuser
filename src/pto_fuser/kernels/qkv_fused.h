#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Fused chunk_o "flash" prototype — score→output in ONE launch (B3).
//
// The chunk_o intra path is q·kᵀ → gate/mask → ·v. After B1 the first two stages
// fuse (gated_qk_native produces the masked score A [M,C,C] and writes it to HBM),
// but the second matmul o = A·v (`nij,nje->nie`, the `o_intra` stage) stays STAGED
// and reads A back from HBM. This kernel folds that second contraction in, so A
// never lands in HBM:
//
//   [Cube]  S = q @ kᵀ           (NT native [M,C,D])        -> S [M,C,C] float
//   [Vec ]  A = S·exp(min(gᵢ-gⱼ,0))·causal   (kkt_epilogue) -> A [M,C,C] half
//   [Cube]  o = A @ v            (NN native [M,C,C]@[M,C,Dv])-> o [M,C,Dv] float
//
// Stage 1 reuses kkt's native-NT config (feed q,k instead of k,k); stage 2 reuses
// `kkt_fused::kkt_epilogue_one` verbatim (gate-kind-agnostic — the host passes
// beta=1 and a causal mask for chunk_o); the only genuinely-new piece is stage 3's
// plain NN contiguous config (`config_av` in the .cpp) and the 3-stage choreography.
//
// ── V1 (this file): two-pass ─────────────────────────────────────────────────
//   Cube produces ALL S tiles to ws_qk [M,C,C], SyncAll, Vec masks ALL to ws_A
//   [M,C,C], SyncAll, Cube contracts ALL o = A·v. Correct + proves the composition
//   and both configs; it still materializes S and A in HBM (no bandwidth win yet —
//   that is the interleaved V2). This is the bit-exact ORACLE for V2.
//
// Operands (host, native [M,C,D] = the Program's own batch, heads outer):
//   q,k   [M,C,D]  half     v [M,C,Dv] half
//   g_t   [M,C]    float    beta_t [M,C] half (=1 for chunk_o)   mask [C,C] float causal
// Output:
//   o     [M,C,Dv] float    (matches the staged o_intra einsum's fp32 output)
// ─────────────────────────────────────────────────────────────────────────────
#include "kkt_fused.h"

namespace qkv_fused {

using namespace pto;

// Vec stage: mask every tile assigned to this core, storing the half score A into
// ws_A (passed as kkt_epilogue_one's "L" — its native path writes [M,C,C] at
// pid*C*C + row_off*C, exactly ws_A's layout). C = n_free0 (score dim, not D), so
// this is correct at C!=D (the zoo) as well as C==D (GDN).
template <typename CFG>
AICORE inline void mask_all(__gm__ const float* ws_qk, __gm__ const float* g_t,
                            __gm__ const half* beta_t, __gm__ const float* mask,
                            __gm__ half* ws_A) {
    constexpr unsigned C = CFG::n_free0;
    constexpr unsigned I = CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned vid = get_subblockid();
    for (unsigned w = 0; w < (I + bnum - 1) / bnum; ++w) {
        unsigned pid = w * bnum + cid;
        if (pid >= I) continue;
        kkt_fused::kkt_epilogue_one<CFG>(ws_qk + (int64_t)pid * C * C, g_t, beta_t,
                                         mask, ws_A, pid, vid);
    }
}

// ── V1: Cube(q@kᵀ) -> SyncAll -> Vec(gate/mask) -> SyncAll -> Cube(A@v) ───────
// QK_CFG: native-NT [M,C,D] score config (kkt's config_einsum, KKT_NATIVE).
// AV_CFG: native-NN [M,C,C]@[M,C,Dv] output config (config_av in the .cpp).
template <typename QK_CFG, typename AV_CFG>
__global__ AICORE void qkv_flash_kernel_v1(__gm__ const half* q, __gm__ const half* k,
                                           __gm__ const half* v, __gm__ const float* g_t,
                                           __gm__ const half* beta_t, __gm__ const float* mask,
                                           __gm__ float* ws_qk, __gm__ half* ws_A,
                                           __gm__ float* o, uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    // Stage 1: S = q @ kᵀ (NT direct-read, both operands native [M,C,D]).
    if constexpr (DAV_CUBE) {
        pto_einsum::batched_matmul_inline<half, QK_CFG, false, false, true>(
            q, k, ws_qk, get_block_idx(), get_block_num());
    }
    pto_einsum::SyncAll<false>();
    // Stage 2: A = gate(S)·mask -> ws_A half.
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        mask_all<QK_CFG>(ws_qk, g_t, beta_t, mask, ws_A);
    }
    pto_einsum::SyncAll<false>();
    // Stage 3: o = A @ v (NN contiguous, A [M,C,C] · v [M,C,Dv]).
    if constexpr (DAV_CUBE) {
        pto_einsum::batched_matmul_inline<half, AV_CFG, false, false, false>(
            ws_A, v, o, get_block_idx(), get_block_num());
    }
}

// ── V2: per-tile interleave, DOUBLE-BUFFERED — S and A never leave L2 ─────────
// FFTS flag ids disjoint from SyncAll (11-14), kkt (0-3), prologue (5-6). Two slots
// each, so S_READY uses {7,8} and A_READY uses {9,10} (base + slot).
constexpr uint16_t QKV_S_READY = 7;   // Cube -> Vec: S tile ready  (+ slot: 7,8)
constexpr uint16_t QKV_A_READY = 9;   // Vec  -> Cube: A tile ready (+ slot: 9,10)
// Two-slot software pipeline hiding the Cube↔Vec handshake: the Cube produces the
// NEXT tile's score S(w+1) into the other slot while the Vec masks the current S(w),
// then consumes o(w)=A(w)·v after the mask. With two slots and the two READY flags,
// the WAR hazards are ordered *for free* by the Cube's own instruction stream — the
// S(w+2) produce (iter w+1) runs strictly after the o(w) consume (iter w), which runs
// strictly after A_READY(w), i.e. after the Vec has read S(w); and the S_READY(w+2)
// signal Vec waits on is only emitted after that same o(w). So no FREE flags are
// needed. S and A live only in the per-core L2 slots — never an [M,C,C] HBM
// materialization (the round-trip flash-V1 pays, and the win capture cannot recover).
//   QK_CFG : batched native-NT score config (tile index pid selects q[pid]/k[pid]).
//   AV_TILE: SINGLE-batch native-NN output config (A from the slot; v[pid]/o[pid] as
//            base pointers). Tail tiles (pid>=I) skip the matmul but keep the flags.
template <typename QK_CFG, typename AV_TILE>
__global__ AICORE void qkv_flash_kernel_v2(__gm__ const half* q, __gm__ const half* k,
                                           __gm__ const half* v, __gm__ const float* g_t,
                                           __gm__ const half* beta_t, __gm__ const float* mask,
                                           __gm__ float* s_ring, __gm__ half* a_ring,
                                           __gm__ float* o, uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = QK_CFG::n_free0;
    constexpr unsigned DV = AV_TILE::n_free1;
    constexpr unsigned I = QK_CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned iters = (I + bnum - 1) / bnum;
    __gm__ float* S0 = s_ring + (int64_t)cid * 2 * C * C;   // 2 [C,C] float slots / core
    __gm__ half*  A0 = a_ring + (int64_t)cid * 2 * C * C;   // 2 [C,C] half  slots / core

    if constexpr (DAV_CUBE) {
        // prologue: produce S(0) so the Vec's first mask can start.
        if (cid < I) {
            pto_einsum::matmul_one_tile_deep<half, QK_CFG, false, true, true>(
                q, k, nullptr, cid, S0);
        }
        ffts_cross_core_sync(PIPE_FIX, pto_einsum::GetffstMsg(0x2, QKV_S_READY + 0));
        for (unsigned w = 0; w < iters; ++w) {
            unsigned slot = w & 1u;
            // Prefetch S(w+1) into the other slot — overlaps the Vec's mask(w).
            if (w + 1 < iters) {
                unsigned nslot = (w + 1) & 1u;
                unsigned npid = (w + 1) * bnum + cid;
                if (npid < I) {
                    pto_einsum::matmul_one_tile_deep<half, QK_CFG, false, true, true>(
                        q, k, nullptr, npid, S0 + (int64_t)nslot * C * C);
                }
                ffts_cross_core_sync(PIPE_FIX,
                                     pto_einsum::GetffstMsg(0x2, QKV_S_READY + nslot));
            }
            // Consume o(w) = A(w) @ v[pid] once the Vec has masked S(w) into A slot.
            unsigned pid = w * bnum + cid;
            wait_flag_dev(QKV_A_READY + slot);
            pipe_barrier(PIPE_ALL);
            if (pid < I) {
                pto_einsum::matmul_one_tile_deep<half, AV_TILE, false, false, true>(
                    A0 + (int64_t)slot * C * C, v + (int64_t)pid * C * DV, nullptr, 0,
                    o + (int64_t)pid * C * DV);
            }
        }
    }
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        for (unsigned w = 0; w < iters; ++w) {
            unsigned slot = w & 1u;
            unsigned pid = w * bnum + cid;
            wait_flag_dev(QKV_S_READY + slot);
            pipe_barrier(PIPE_ALL);
            // A = gate(S)·mask -> A slot. Passing (A_slot - pid*C*C) as the "L" base
            // cancels kkt_epilogue_one's native pid*C*C store offset, so the tile lands
            // in the per-core slot rather than a [M,C,C] position.
            if (pid < I) {
                __gm__ half* A_slot = A0 + (int64_t)slot * C * C;
                kkt_fused::kkt_epilogue_one<QK_CFG>(
                    S0 + (int64_t)slot * C * C, g_t, beta_t, mask,
                    A_slot - (int64_t)pid * C * C, pid, vid);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, QKV_A_READY + slot));
        }
    }
}

} // namespace qkv_fused
