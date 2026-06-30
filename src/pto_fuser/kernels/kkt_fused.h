#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Fused GDN `kkt` prototype.
//
// Demonstrates the auto-fuser thesis: drive the Cube contraction with the EINSUM
// tile-matmul core (`matmul_one_tile_deep`, NT direct-read — no Phase A), then
// replace Phase C with a *gated-mask epilogue* that applies the GDN decay/beta/
// causal-mask ON-CHIP, instead of as torch HBM glue. One mix-kernel launch.
//
//   [Cube]  qk = k @ k^T  (NT: both operands read straight from k, contraction
//           innermost) -> [C,C] float
//   [Vec ]  L[i,j] = qk[i,j] * exp(min(g_v[i]-g[j], 0)) * (i>j),
//           g_v[i] = g[i] + log(beta[i])                    -> L [1,T,H,C] half
//
// ── V1 (kkt_fused_kernel): two-pass ──────────────────────────────────────────
//   Cube produces ALL tiles to ws_res [I,C,C] in HBM, ONE SyncAll<false>(), then
//   Vec consumes ALL tiles. Isolates the glue-elimination win, but pays a full qk
//   HBM round-trip (write I*C*C + read I*C*C). I = nc*H.
//
// ── V2 (kkt_fused_kernel_v2): per-tile interleave ────────────────────────────
//   Per-core Cube<->Vec FFTS ping-pong over a TINY ring buffer ws_ping
//   [block_num * 2, C, C] (~2 slots/core, L2-resident — never a full I*C*C HBM
//   materialization). Cube produces tile t into slot (cid*2 + (t&1)), signals its
//   paired Vec; Vec consumes on-chip and signals the slot free. This removes the
//   qk HBM round-trip — the only residual vs megagdn's hand-written kernel.
//   (Mirrors megagdn/scaled_dot_kkt.cpp's protocol, but reuses the einsum matmul
//   core via the EXT_STORE hook instead of an open-coded GEMM.)
//
// Inputs (host-prepared to match the bench's kkt operands):
//   k     [nc*C, H, D] half  — GQA-expanded keys, contiguous BSND (token,h,d)
//   g_t   [H, T]       float — cumulative gate, transposed to head-major
//   beta_t[H, T]       half  — beta, head-major
//   mask  [C, C]       float — strict lower-triangular (i>j)
// Output:
//   L     [T, H, C]    half  — gated intra-chunk matrix, BSND
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_einsum.h"

namespace kkt_fused {

using namespace pto;

// Ping-pong cross-core flag ids (disjoint from SyncAll's 11-14, so the two
// protocols never alias). Cube signals 0/1 "tile ready"; Vec signals 2/3 "slot
// free". mode 2 == broadcast within the AICore group (cube <-> its paired Vec).
constexpr uint16_t PP_READY = 0;   // + slot
constexpr uint16_t PP_FREE  = 2;   // + slot

// Intra-Vec RAW barrier in the gated epilogue. KKT_NOBARRIER compiles them out
// (numerically WRONG — every epilogue op is a RAW on the previous; timing-only
// knob to isolate the full-pipe-drain cost of the barriers from the op latency).
#ifdef KKT_NOBARRIER
#define VBAR()
#else
#define VBAR() pipe_barrier(PIPE_V)
#endif

// One (work-item, half) sub-block of the gated epilogue: HalfC rows x C cols.
// Reads the qk tile from `qk_base` (a [C,C] float tile: V1 -> ws_res+pid*C*C,
// V2 -> the ring slot), gates + masks, stores L. `pid` indexes (chunk,head).
template <typename CFG>
AICORE inline void kkt_epilogue_one(__gm__ const float* qk_base, __gm__ const float* g_t,
                                    __gm__ const half* beta_t, __gm__ const float* mask,
                                    __gm__ half* L, unsigned pid, unsigned vid) {
    constexpr unsigned C = CFG::n_contract;          // 128
    constexpr unsigned HalfC = C / 2;                // 64
    constexpr int32_t Hh = CFG::kkt_H;
    constexpr int64_t Tt = CFG::kkt_T;

    // UB byte map (HalfC x C float tiles = 32 KiB each).
    constexpr unsigned QKF  = 0;
    constexpr unsigned COEF = QKF  + HalfC * C * 4;
    constexpr unsigned MSK  = COEF + HalfC * C * 4;
    constexpr unsigned GR2D = MSK  + HalfC * C * 4;
    constexpr unsigned GCOL = GR2D + HalfC * C * 4;            // [1,C] f32
    constexpr unsigned GROW = GCOL + C * 4;                    // [1,HalfC] f32
    constexpr unsigned BETF = GROW + HalfC * 4;                // [1,HalfC] f32
    constexpr unsigned BETH = BETF + HalfC * 4;                // [1,HalfC] half

    unsigned row_off = vid * HalfC;
    unsigned chunk = pid / (unsigned)Hh;
    unsigned head  = pid % (unsigned)Hh;
    int64_t tok0 = (int64_t)chunk * C;
    int64_t g_base = (int64_t)head * Tt + tok0;

    using TileF   = Tile<TileType::Vec, float, HalfC, C, BLayout::RowMajor, -1, -1>;
    using TileGc  = Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, -1, -1>;
    using TileGr  = Tile<TileType::Vec, float, 1, HalfC, BLayout::RowMajor, -1, -1>;
    using TileGvC = Tile<TileType::Vec, float, HalfC, 1, BLayout::ColMajor, -1, -1>;
    using TileBh  = Tile<TileType::Vec, half,  1, HalfC, BLayout::RowMajor, -1, -1>;
    using TileHout= Tile<TileType::Vec, half,  HalfC, C, BLayout::RowMajor, -1, -1>;

    using GmF   = GlobalTensor<float, Shape<1,1,1,-1,C>, Stride<1,1,1,C,1>>;
    using GmGc  = GlobalTensor<float, Shape<1,1,1,1,C>,  Stride<1,1,1,C,1>>;
    using GmGr  = GlobalTensor<float, Shape<1,1,1,1,HalfC>, Stride<1,1,1,HalfC,1>>;
    using GmBh  = GlobalTensor<half,  Shape<1,1,1,1,HalfC>, Stride<1,1,1,HalfC,1>>;
    using GmL   = GlobalTensor<half,  Shape<1,1,1,-1,C>, Stride<1,1,1,(int64_t)Hh*C,1>>;

    // ── loads ───────────────────────────────────────────────────────────────
    TileF qkf;  TASSIGN(qkf, QKF);   qkf.SetValidRow(HalfC); qkf.SetValidCol(C);
    GmF gqk(const_cast<__gm__ float*>(qk_base + (int64_t)row_off * C), Shape<1,1,1,-1,C>(HalfC));
    TLOAD(qkf, gqk);

#ifdef KKT_PLAIN
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TileHout outp; TASSIGN(outp, GR2D); outp.SetValidRow(HalfC); outp.SetValidCol(C);
    TCVT(outp, qkf, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    int64_t l_offp = ((tok0 + row_off) * (int64_t)Hh + head) * C;
    GmL glp(L + l_offp, Shape<1,1,1,-1,C>(HalfC));
    TSTORE(glp, outp);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    return;
#endif
    TileGc gcol; TASSIGN(gcol, GCOL); gcol.SetValidRow(1); gcol.SetValidCol(C);
    GmGc ggc(const_cast<__gm__ float*>(g_t + g_base));
    TLOAD(gcol, ggc);

    TileGr grow; TASSIGN(grow, GROW); grow.SetValidRow(1); grow.SetValidCol(HalfC);
    GmGr ggr(const_cast<__gm__ float*>(g_t + g_base + row_off));
    TLOAD(grow, ggr);

    TileBh bh; TASSIGN(bh, BETH); bh.SetValidRow(1); bh.SetValidCol(HalfC);
    GmBh gbh(const_cast<__gm__ half*>(beta_t + g_base + row_off));
    TLOAD(bh, gbh);

    TileF msk; TASSIGN(msk, MSK); msk.SetValidRow(HalfC); msk.SetValidCol(C);
#ifndef KKT_NOMASK
    GmF gmsk(const_cast<__gm__ float*>(mask + (int64_t)row_off * C), Shape<1,1,1,-1,C>(HalfC));
    TLOAD(msk, gmsk);
#endif

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ── g_v[i] = g_row[i] + log(beta[i]) ─────────────────────────────────────
    TileGr betf; TASSIGN(betf, BETF); betf.SetValidRow(1); betf.SetValidCol(HalfC);
    TCVT(betf, bh, RoundMode::CAST_RINT);
    VBAR();
    TLOG(betf, betf);
    VBAR();
    TileGr gv; TASSIGN(gv, GROW);  gv.SetValidRow(1); gv.SetValidCol(HalfC);  // reuse GROW
    TADD(gv, grow, betf);
    VBAR();

    // ── coeff[i,j] = exp(min(g_v[i] - g[j], 0)) ──────────────────────────────
    // g_v lives at GROW (row-major [1,HalfC]); alias it as a [HalfC,1] col-major
    // source for the row-broadcast (same bytes — a length-HalfC vector).
    TileGvC gvc; TASSIGN(gvc, GROW); gvc.SetValidRow(HalfC); gvc.SetValidCol(1);
    TileF gr2d; TASSIGN(gr2d, GR2D); gr2d.SetValidRow(HalfC); gr2d.SetValidCol(C);
    TROWEXPAND(gr2d, gvc);                 // gr2d[i,j] = g_v[i]
    VBAR();
    TileF coef; TASSIGN(coef, COEF); coef.SetValidRow(HalfC); coef.SetValidCol(C);
    TCOLEXPANDSUB(coef, gr2d, gcol);       // coef[i,j] = g_v[i] - g[j]
    VBAR();
    TMINS(coef, coef, 0.0f);
    VBAR();
    TEXP(coef, coef);
    VBAR();

    // ── L = qk * coeff * mask ────────────────────────────────────────────────
    TMUL(qkf, qkf, coef);
    VBAR();
#ifndef KKT_NOMASK
    TMUL(qkf, qkf, msk);
    VBAR();
#endif
    // GR2D region (HalfC*C*4 bytes) is free now; reuse it for the half output tile.
    TileHout outh2; TASSIGN(outh2, GR2D); outh2.SetValidRow(HalfC); outh2.SetValidCol(C);
    TCVT(outh2, qkf, RoundMode::CAST_RINT);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    int64_t l_off = ((tok0 + row_off) * (int64_t)Hh + head) * C;
    GmL gl(L + l_off, Shape<1,1,1,-1,C>(HalfC));
    TSTORE(gl, outh2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

// ── V1 Vec epilogue: loop all tiles assigned to this core ───────────────────
template <typename CFG>
AICORE inline void kkt_epilogue(__gm__ const float* ws_res, __gm__ const float* g_t,
                                __gm__ const half* beta_t, __gm__ const float* mask,
                                __gm__ half* L) {
    constexpr unsigned C = CFG::n_contract;
    constexpr unsigned I = CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned vid = get_subblockid();                 // 0/1 -> upper/lower half
    for (unsigned w = 0; w < (I + bnum - 1) / bnum; ++w) {
        unsigned pid = w * bnum + cid;
        if (pid >= I) continue;
        kkt_epilogue_one<CFG>(ws_res + (int64_t)pid * C * C, g_t, beta_t, mask, L, pid, vid);
    }
}

template <typename CFG>
__global__ AICORE void kkt_fused_kernel(__gm__ const half* k, __gm__ const float* g_t,
                                        __gm__ const half* beta_t, __gm__ const float* mask,
                                        __gm__ float* ws_res, __gm__ half* L,
                                        uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    if constexpr (DAV_CUBE) {
        pto_einsum::batched_matmul_inline<half, CFG, false, false, true>(
            k, k, ws_res, get_block_idx(), get_block_num());
    }
    pto_einsum::SyncAll<false>();
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        kkt_epilogue<CFG>(ws_res, g_t, beta_t, mask, L);
    }
}

// ── V2: per-tile interleave via per-core Cube<->Vec FFTS ping-pong ───────────
// ws_ping is a [block_num*2, C, C] float ring. Each core owns 2 slots
// (cid*2 + slot); Cube fills slot (t&1) and signals PP_READY+slot, Vec drains it
// and signals PP_FREE+slot. No global SyncAll — production and consumption stream
// per tile, so the qk working set is 2 slots/core (L2), never I*C*C in HBM.
template <typename CFG>
__global__ AICORE void kkt_fused_kernel_v2(__gm__ const half* k, __gm__ const float* g_t,
                                           __gm__ const half* beta_t, __gm__ const float* mask,
                                           __gm__ float* ws_ping, __gm__ half* L,
                                           uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = CFG::n_contract;
    constexpr unsigned I = CFG::n_inplace;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();
    unsigned iters = (I + bnum - 1) / bnum;

    if constexpr (DAV_CUBE) {
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            unsigned slot = w & 1u;
            // Wait this slot free (Vec released it / primed at start), then produce
            // tile pid into it and signal it ready.
            wait_flag_dev(PP_FREE + slot);
            pipe_barrier(PIPE_ALL);
            if (pid < I) {
                __gm__ float* dst = ws_ping + (int64_t)(cid * 2u + slot) * C * C;
#ifndef KKT_VEC_NOMATMUL
                pto_einsum::matmul_one_tile_deep<half, CFG, false, true, true>(
                    k, k, nullptr, pid, dst);
#endif
            }
            ffts_cross_core_sync(PIPE_FIX, pto_einsum::GetffstMsg(0x2, PP_READY + slot));
        }
    }
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        unsigned vid = get_subblockid();
        // Prime: both slots free so Cube's first two produces can start.
        ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, PP_FREE + 0));
        ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, PP_FREE + 1));
        for (unsigned w = 0; w < iters; ++w) {
            unsigned pid = w * bnum + cid;
            unsigned slot = w & 1u;
            wait_flag_dev(PP_READY + slot);
            pipe_barrier(PIPE_ALL);
#ifndef KKT_CUBE_ONLY
            if (pid < I) {
                __gm__ const float* src = ws_ping + (int64_t)(cid * 2u + slot) * C * C;
                kkt_epilogue_one<CFG>(src, g_t, beta_t, mask, L, pid, vid);
            }
#endif
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, PP_FREE + slot));
        }
    }
}

} // namespace kkt_fused
