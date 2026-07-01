#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Per-dim PROLOGUE prototype — the per-channel-gate counterpart of kkt's epilogue.
//
// For a SCALAR gate the intra-chunk score A = tril((q⊙P)(k⊙1/P)ᵀ) factors as
// (q kᵀ)·exp(gᵢ-gⱼ)·causal, so the decay rides OUT of the contraction into the Vec
// *epilogue* (kkt_fused.h). For a PER-CHANNEL gate (GLA, KDA) P_id lives INSIDE Σ_d
// — no scalar coeff reproduces it — so the decay must be applied to the *operands*
// before the contraction: a Vec PROLOGUE q̃ = q⊙P, k̂ = k⊙(1/P), then the plain
// matmul, then only the causal/tril mask.
//
// V1 (this file): three sequential passes in ONE launch (dispatch-elim), reusing the
// einsum matmul core for the contraction.
//   [Vec ]  q̃ = q⊙P, k̂ = k⊙invP            -> qd_s, kinv_s [M,C,D] (kernel scratch)
//   SyncAll
//   [Cube]  A = q̃ @ k̂ᵀ  (NT, native [M,C,D]) -> ws_res [M,C,C]
//   SyncAll
//   [Vec ]  L = tril(A)·causal                -> L [M,C,C] half
// This proves the partition + correctness; qd_s/kinv_s still round-trip the scratch,
// so at captured (dispatch already gone) it is ~parity with staged — the bandwidth
// win needs the operands scaled in L1/UB (the bespoke L2-resident matmul, V2 follow-up).
//
// Operands (host, native [M,C,D] = the Program's own batch, heads outer):
//   q, k          [M,C,D] half     P, invP [M,C,D] half  (the cumprod decay + inverse)
//   mask          [C,C]   float    causal (i>=j)
// Output:
//   L             [M,C,C] half
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_einsum.h"

namespace prologue_fused {

using namespace pto;

// One (work-item, half) sub-block of the operand prescale: HalfC rows x D cols.
// Scales q⊙P and k⊙invP for batch `pid`, sub-block `vid` (upper/lower HalfC rows).
template <typename CFG>
AICORE inline void prescale_one(__gm__ const half* q, __gm__ const half* P,
                                __gm__ const half* k, __gm__ const half* invP,
                                __gm__ half* qd_s, __gm__ half* kinv_s,
                                unsigned pid, unsigned vid) {
    constexpr unsigned C = CFG::n_free0;
    constexpr unsigned D = CFG::n_contract;
    constexpr unsigned HalfC = C / 2;
    unsigned row_off = vid * HalfC;
    int64_t base = (int64_t)pid * C * D + (int64_t)row_off * D;   // [M,C,D] contiguous

    using TileH = Tile<TileType::Vec, half, HalfC, D, BLayout::RowMajor, -1, -1>;
    using GmH   = GlobalTensor<half, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    constexpr unsigned AT = 0;
    constexpr unsigned BT = HalfC * D * 2;                        // half bytes

    // q̃ = q ⊙ P
    TileH a; TASSIGN(a, AT); a.SetValidRow(HalfC); a.SetValidCol(D);
    TileH b; TASSIGN(b, BT); b.SetValidRow(HalfC); b.SetValidCol(D);
    GmH gq(const_cast<__gm__ half*>(q + base), Shape<1,1,1,-1,D>(HalfC)); TLOAD(a, gq);
    GmH gp(const_cast<__gm__ half*>(P + base), Shape<1,1,1,-1,D>(HalfC)); TLOAD(b, gp);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(a, a, b);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GmH gqd(qd_s + base, Shape<1,1,1,-1,D>(HalfC)); TSTORE(gqd, a);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // k̂ = k ⊙ invP
    TileH c; TASSIGN(c, AT); c.SetValidRow(HalfC); c.SetValidCol(D);
    TileH d; TASSIGN(d, BT); d.SetValidRow(HalfC); d.SetValidCol(D);
    GmH gk(const_cast<__gm__ half*>(k + base), Shape<1,1,1,-1,D>(HalfC)); TLOAD(c, gk);
    GmH gi(const_cast<__gm__ half*>(invP + base), Shape<1,1,1,-1,D>(HalfC)); TLOAD(d, gi);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(c, c, d);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GmH gki(kinv_s + base, Shape<1,1,1,-1,D>(HalfC)); TSTORE(gki, c);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// One sub-block of the causal-mask epilogue: HalfC rows x C cols. L = tril(A)·mask.
template <typename CFG>
AICORE inline void mask_one(__gm__ const float* qk_base, __gm__ const float* mask,
                            __gm__ half* L, unsigned pid, unsigned vid) {
    constexpr unsigned C = CFG::n_free0;
    constexpr unsigned HalfC = C / 2;
    unsigned row_off = vid * HalfC;

    using TileF    = Tile<TileType::Vec, float, HalfC, C, BLayout::RowMajor, -1, -1>;
    using TileHout = Tile<TileType::Vec, half,  HalfC, C, BLayout::RowMajor, -1, -1>;
    using GmF = GlobalTensor<float, Shape<1,1,1,-1,C>, Stride<1,1,1,C,1>>;
    using GmL = GlobalTensor<half,  Shape<1,1,1,-1,C>, Stride<1,1,1,C,1>>;
    constexpr unsigned QKF = 0;
    constexpr unsigned MSK = QKF + HalfC * C * 4;
    constexpr unsigned OUT = MSK + HalfC * C * 4;

    TileF qkf; TASSIGN(qkf, QKF); qkf.SetValidRow(HalfC); qkf.SetValidCol(C);
    GmF gqk(const_cast<__gm__ float*>(qk_base + (int64_t)row_off * C), Shape<1,1,1,-1,C>(HalfC));
    TLOAD(qkf, gqk);
    TileF msk; TASSIGN(msk, MSK); msk.SetValidRow(HalfC); msk.SetValidCol(C);
    GmF gmsk(const_cast<__gm__ float*>(mask + (int64_t)row_off * C), Shape<1,1,1,-1,C>(HalfC));
    TLOAD(msk, gmsk);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(qkf, qkf, msk);
    pipe_barrier(PIPE_V);
    TileHout outh; TASSIGN(outh, OUT); outh.SetValidRow(HalfC); outh.SetValidCol(C);
    TCVT(outh, qkf, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    int64_t l_off = (int64_t)pid * C * C + (int64_t)row_off * C;
    GmL gl(L + l_off, Shape<1,1,1,-1,C>(HalfC)); TSTORE(gl, outh);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// V1: Vec prescale -> SyncAll -> Cube matmul -> SyncAll -> Vec mask, one launch.
template <typename CFG>
__global__ AICORE void qk_prologue_kernel(__gm__ const half* q, __gm__ const half* P,
                                          __gm__ const half* k, __gm__ const half* invP,
                                          __gm__ const float* mask,
                                          __gm__ half* qd_s, __gm__ half* kinv_s,
                                          __gm__ float* ws_res, __gm__ half* L,
                                          uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = CFG::n_free0;
    constexpr unsigned I = CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned iters = (I + bnum - 1) / bnum;

    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            if (pid < I) prescale_one<CFG>(q, P, k, invP, qd_s, kinv_s, pid, vid);
        }
    }
    pto_einsum::SyncAll<false>();
    if constexpr (DAV_CUBE) {
        pto_einsum::batched_matmul_inline<half, CFG, false, false, true>(
            qd_s, kinv_s, ws_res, get_block_idx(), get_block_num());
    }
    pto_einsum::SyncAll<false>();
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            if (pid < I) mask_one<CFG>(ws_res + (int64_t)pid * C * C, mask, L, pid, vid);
        }
    }
}

// One sub-block prescale writing into a per-core RING SLOT (offset row_off*D within
// the slot) instead of the full [M,C,D] scratch — the V2 path so qd/kinv stay L2.
template <typename CFG>
AICORE inline void prescale_to_slot(__gm__ const half* q, __gm__ const half* P,
                                    __gm__ const half* k, __gm__ const half* invP,
                                    __gm__ half* opA, __gm__ half* opB,
                                    unsigned pid, unsigned vid) {
    constexpr unsigned C = CFG::n_free0;
    constexpr unsigned D = CFG::n_contract;
    constexpr unsigned HalfC = C / 2;
    unsigned row_off = vid * HalfC;
    int64_t src = (int64_t)pid * C * D + (int64_t)row_off * D;
    int64_t dst = (int64_t)row_off * D;

    using TileH = Tile<TileType::Vec, half, HalfC, D, BLayout::RowMajor, -1, -1>;
    using GmH   = GlobalTensor<half, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    TileH a; TASSIGN(a, 0); a.SetValidRow(HalfC); a.SetValidCol(D);
    TileH b; TASSIGN(b, HalfC * D * 2); b.SetValidRow(HalfC); b.SetValidCol(D);
    GmH gq(const_cast<__gm__ half*>(q + src), Shape<1,1,1,-1,D>(HalfC)); TLOAD(a, gq);
    GmH gp(const_cast<__gm__ half*>(P + src), Shape<1,1,1,-1,D>(HalfC)); TLOAD(b, gp);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(a, a, b);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GmH gqd(opA + dst, Shape<1,1,1,-1,D>(HalfC)); TSTORE(gqd, a);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    TileH c; TASSIGN(c, 0); c.SetValidRow(HalfC); c.SetValidCol(D);
    TileH d; TASSIGN(d, HalfC * D * 2); d.SetValidRow(HalfC); d.SetValidCol(D);
    GmH gk(const_cast<__gm__ half*>(k + src), Shape<1,1,1,-1,D>(HalfC)); TLOAD(c, gk);
    GmH gi(const_cast<__gm__ half*>(invP + src), Shape<1,1,1,-1,D>(HalfC)); TLOAD(d, gi);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(c, c, d);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GmH gki(opB + dst, Shape<1,1,1,-1,D>(HalfC)); TSTORE(gki, c);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// FFTS flag ids for the V2 Vec<->Cube ping-pong (disjoint from SyncAll 11-14 and kkt's
// 0-3). Vec signals operands-ready; Cube signals A-tile-ready.
constexpr uint16_t PRO_OP_READY = 5;   // Vec -> Cube: opA/opB filled
constexpr uint16_t PRO_A_READY  = 6;   // Cube -> Vec: Aout filled

// ── V2: per-tile Vec(prescale) -> Cube(matmul from L2 ring) -> Vec(mask) ──────
// The scaled operands live in a per-core L2-resident ring (opA/opB slots), never the
// full [M,C,D] HBM scratch — matmul_one_tile_deep<TILE_CFG> reads each operand from the
// slot (single-tile config, base offset 0) and EXT_STOREs the A tile to the Aout slot,
// which Vec masks on-chip. Sequential single-slot ping-pong (correct; cross-tile overlap
// is a later tuning). TILE_CFG is the single-batch [C,D]@[C,D]ᵀ config (n_inplace=1).
template <typename CFG, typename TILE_CFG>
__global__ AICORE void qk_prologue_kernel_v2(__gm__ const half* q, __gm__ const half* P,
                                             __gm__ const half* k, __gm__ const half* invP,
                                             __gm__ const float* mask,
                                             __gm__ half* op_ring, __gm__ float* a_ring,
                                             __gm__ half* L, uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = CFG::n_free0;
    constexpr unsigned D = CFG::n_contract;
    constexpr unsigned I = CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned iters = (I + bnum - 1) / bnum;
    // Per-core slots: opA|opB (C*D halfs each) in op_ring, one A tile (C*C floats) in a_ring.
    __gm__ half*  opA  = op_ring + (int64_t)cid * 2 * C * D;
    __gm__ half*  opB  = opA + (int64_t)C * D;
    __gm__ float* Aout = a_ring + (int64_t)cid * C * C;

    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            if (pid < I) prescale_to_slot<CFG>(q, P, k, invP, opA, opB, pid, vid);
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, PRO_OP_READY));
            wait_flag_dev(PRO_A_READY);
            pipe_barrier(PIPE_ALL);
            if (pid < I) mask_one<CFG>(Aout, mask, L, pid, vid);
        }
    }
    if constexpr (DAV_CUBE) {
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            wait_flag_dev(PRO_OP_READY);
            pipe_barrier(PIPE_ALL);
#ifndef PRO_VEC_NOMATMUL
            if (pid < I) {
                pto_einsum::matmul_one_tile_deep<half, TILE_CFG, false, true, true>(
                    opA, opB, nullptr, 0, Aout);
            }
#endif
            ffts_cross_core_sync(PIPE_FIX, pto_einsum::GetffstMsg(0x2, PRO_A_READY));
        }
    }
}

} // namespace prologue_fused
