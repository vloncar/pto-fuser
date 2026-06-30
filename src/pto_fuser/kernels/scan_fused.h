#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// T2 cross-chunk-scan prototype (GDN chunk_h topology, simplified).
//
// Tests the genuinely-new chunk-attention dataflow: a SEQUENTIAL recurrence over
// chunks where the resident state S is BOTH a matmul operand (w@S) and a
// Vec-updated accumulator (S = decay*S + k^T@vc). Parallel over (b,h) across
// cores; sequential over chunks within each (b,h). The two GEMMs reuse the EINSUM
// tile-matmul core (`matmul_one_tile_deep`) via its two landed strided read modes:
//   w @ S      -> in_nt=2 (NN-strided: A=w natural, B=S strided-K)
//   k^T @ vc   -> in_nt=3 (TN: A=k transposed via DN, B=vc NN-strided)
//
// Per (b,h), sequential over c (S resident, stored half like megagdn chunk_h):
//   h_out[c] = S
//   WS = w[c] @ S                 [Cube, in_nt=2 -> WS f32]
//   vc = u[c] - WS                [Vec]
//   S  = decay[c]*S + k[c]^T@vc   [Cube in_nt=3 -> kv f32; Vec recurrence in f32]
//
// ── V0 (this file): correctness + schedule skeleton ──────────────────────────
//   S/WS/vc/kv all round-trip through a per-core GM workspace (NOT resident yet).
//   Proves the composition is bit-exact and the parallel-(b,h)/sequential-chunk
//   schedule works in ONE kernel launch. V1 will keep S resident on-chip.
//
// Cube<->Vec handshake: strictly-serial 4-flag FFTS protocol (no global SyncAll).
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_einsum.h"

namespace chunk_h_scan {

using namespace pto;

constexpr uint16_t F_MM1 = 0;   // Cube -> Vec : WS = w@S ready
constexpr uint16_t F_RES = 1;   // Vec -> Cube : vc = u-WS ready
constexpr uint16_t F_MM2 = 2;   // Cube -> Vec : kv = k^T@vc ready
constexpr uint16_t F_REC = 3;   // Vec -> Cube : S updated (or primed at unit start)

// Per-core workspace byte offsets. S is half (matmul operand); WS/KV are f32 (matmul out).
template <typename P>
struct WsMap {
    static constexpr unsigned C = P::C, D = P::D;
    static constexpr unsigned S_OFF  = 0;                  // [D,D] half resident state
    static constexpr unsigned WS_OFF = S_OFF  + D * D * 2; // [C,D] f32 matmul-1 out
    static constexpr unsigned KV_OFF = WS_OFF + C * D * 4; // [D,D] f32 matmul-2 out
    static constexpr unsigned VC_OFF = KV_OFF + D * D * 4; // [C,D] half matmul-2 in
    static constexpr unsigned BYTES  = VC_OFF + C * D * 2;
};

// ── Vec: zero the resident state at the start of a (b,h) unit (load x; x-x; store) ─
template <typename P>
AICORE inline void vec_zero_state(__gm__ half* S_gm) {
    constexpr unsigned D = P::D;
    using TileH = Tile<TileType::Vec, half, D, D, BLayout::RowMajor, -1, -1>;
    using GmH   = GlobalTensor<half, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    TileH s; TASSIGN(s, 0); s.SetValidRow(D); s.SetValidCol(D);
    GmH g(S_gm, Shape<1,1,1,-1,D>(D));
    TLOAD(s, g);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUB(s, s, s);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(g, s);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

// ── Vec: readout h_out[c] = S, then vc = u[c] - WS ───────────────────────────
template <typename P>
AICORE inline void vec_readout_residual(__gm__ const half* u, __gm__ const float* WS_gm,
                                        __gm__ const half* S_gm, __gm__ half* hout_gm,
                                        __gm__ half* vc_gm, int64_t u_off) {
    constexpr unsigned C = P::C, D = P::D, H = P::H;
    using TileHd = Tile<TileType::Vec, half,  D, D, BLayout::RowMajor, -1, -1>;
    using TileFc = Tile<TileType::Vec, float, C, D, BLayout::RowMajor, -1, -1>;
    using TileHc = Tile<TileType::Vec, half,  C, D, BLayout::RowMajor, -1, -1>;
    using GmHd  = GlobalTensor<half,  Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    using GmFc  = GlobalTensor<float, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    using GmHcS = GlobalTensor<half,  Shape<1,1,1,-1,D>, Stride<1,1,1,(int64_t)H*D,1>>; // strided u
    using GmHc  = GlobalTensor<half,  Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;            // contig vc

    // readout: S(half) -> h_out(half), direct copy
    TileHd s; TASSIGN(s, 0); s.SetValidRow(D); s.SetValidCol(D);
    GmHd gs(const_cast<__gm__ half*>(S_gm), Shape<1,1,1,-1,D>(D));
    TLOAD(s, gs);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    GmHd gh(hout_gm, Shape<1,1,1,-1,D>(D));
    TSTORE(gh, s);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);

    // residual: vc = u - WS (f32 math, cast to half). Reuse UB from offset 0.
    TileFc ws; TASSIGN(ws, 0);          ws.SetValidRow(C); ws.SetValidCol(D);
    GmFc gw(const_cast<__gm__ float*>(WS_gm), Shape<1,1,1,-1,D>(C));
    TLOAD(ws, gw);
    TileHc uh; TASSIGN(uh, C*D*4);      uh.SetValidRow(C); uh.SetValidCol(D);
    GmHcS gu(const_cast<__gm__ half*>(u + u_off), Shape<1,1,1,-1,D>(C));
    TLOAD(uh, gu);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TileFc uf; TASSIGN(uf, C*D*4 + C*D*2);  uf.SetValidRow(C); uf.SetValidCol(D);
    TCVT(uf, uh, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    TSUB(uf, uf, ws);
    pipe_barrier(PIPE_V);
    TileHc vch; TASSIGN(vch, C*D*4);    vch.SetValidRow(C); vch.SetValidCol(D);   // reuse uh region
    TCVT(vch, uf, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    GmHc gv(vc_gm, Shape<1,1,1,-1,D>(C));
    TSTORE(gv, vch);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
}

// ── Vec: recurrence S = decay*S + kv (S half; accumulate in f32) ──────────────
template <typename P>
AICORE inline void vec_recurrence(__gm__ half* S_gm, __gm__ const float* KV_gm, float dval) {
    constexpr unsigned D = P::D;
    using TileH = Tile<TileType::Vec, half,  D, D, BLayout::RowMajor, -1, -1>;
    using TileF = Tile<TileType::Vec, float, D, D, BLayout::RowMajor, -1, -1>;
    using GmH   = GlobalTensor<half,  Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    using GmF   = GlobalTensor<float, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    TileH sh; TASSIGN(sh, 0);       sh.SetValidRow(D); sh.SetValidCol(D);
    GmH gs(S_gm, Shape<1,1,1,-1,D>(D));
    TLOAD(sh, gs);
    TileF kv; TASSIGN(kv, D*D*2 + D*D*4); kv.SetValidRow(D); kv.SetValidCol(D);
    GmF gk(const_cast<__gm__ float*>(KV_gm), Shape<1,1,1,-1,D>(D));
    TLOAD(kv, gk);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TileF sf; TASSIGN(sf, D*D*2);   sf.SetValidRow(D); sf.SetValidCol(D);
    TCVT(sf, sh, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    TMULS(sf, sf, dval);
    pipe_barrier(PIPE_V);
    TADD(sf, sf, kv);
    pipe_barrier(PIPE_V);
    TCVT(sh, sf, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gs, sh);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

// ── Vec: per-dim recurrence S = diag(dvec)·S + kv (dvec[k] scales S row k) ─────
// KDA's cross-chunk decay is a per-dimension vector (one rate per K-row of S),
// unlike GDN's single scalar. The row-scale is a single fused TROWEXPANDMUL
// (dvec[D] expanded down the V columns and multiplied into S in one instruction),
// in full f32 — no materialized [D,D] broadcast tile, so sh/sf/kv each keep their
// own UB slot, kv loads up front, and there is no expand-then-mul WAR edge. This
// halves the per-dim Vec work vs the old TROWEXPAND+TMUL pair (the large-H lever).
template <typename P>
AICORE inline void vec_recurrence_perdim(__gm__ half* S_gm, __gm__ const float* KV_gm,
                                         __gm__ const float* dvec) {
    constexpr unsigned D = P::D;
    using TileH  = Tile<TileType::Vec, half,  D, D, BLayout::RowMajor, -1, -1>;
    using TileF  = Tile<TileType::Vec, float, D, D, BLayout::RowMajor, -1, -1>;
    using TileDr = Tile<TileType::Vec, float, 1, D, BLayout::RowMajor, -1, -1>;  // decay [1,D] load
    using TileDc = Tile<TileType::Vec, float, D, 1, BLayout::ColMajor, -1, -1>;  // [D,1] alias (same bytes)
    using GmH    = GlobalTensor<half,  Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    using GmF    = GlobalTensor<float, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    using GmDr   = GlobalTensor<float, Shape<1,1,1,1,-1>, Stride<1,1,1,D,1>>;
    // resident S(half) -> sf(float); kv and the per-dim decay vector loaded up front,
    // each in its own slot (no slot reuse -> no expand/mul WAR edge to insure).
    TileH sh; TASSIGN(sh, 0);       sh.SetValidRow(D); sh.SetValidCol(D);
    GmH gs(S_gm, Shape<1,1,1,-1,D>(D));
    TLOAD(sh, gs);
    TileF kv; TASSIGN(kv, D*D*2 + D*D*4);  kv.SetValidRow(D); kv.SetValidCol(D);
    GmF gk(const_cast<__gm__ float*>(KV_gm), Shape<1,1,1,-1,D>(D));
    TLOAD(kv, gk);
    TileDr dr; TASSIGN(dr, D*D*2 + D*D*4 + D*D*4);  dr.SetValidRow(1); dr.SetValidCol(D);
    GmDr gd(const_cast<__gm__ float*>(dvec), Shape<1,1,1,1,-1>(D));
    TLOAD(dr, gd);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TileF sf; TASSIGN(sf, D*D*2);   sf.SetValidRow(D); sf.SetValidCol(D);
    TCVT(sf, sh, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    // per-dim row-scale in ONE fused op: sf[i,j] *= dvec[i]. dc aliases dr's bytes as a
    // [D,1] col-major vector (TROWEXPANDMUL's expand operand); no [D,D] broadcast tile.
    TileDc dc; TASSIGN(dc, D*D*2 + D*D*4 + D*D*4);  dc.SetValidRow(D); dc.SetValidCol(1);
    TROWEXPANDMUL(sf, sf, dc);
    pipe_barrier(PIPE_V);
    TADD(sf, sf, kv);
    pipe_barrier(PIPE_V);
    TCVT(sh, sf, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gs, sh);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

// ── Vec: store final state final[b,h] = S ────────────────────────────────────
template <typename P>
AICORE inline void vec_store_final(__gm__ const half* S_gm, __gm__ half* final_gm) {
    constexpr unsigned D = P::D;
    using TileH = Tile<TileType::Vec, half, D, D, BLayout::RowMajor, -1, -1>;
    using GmH   = GlobalTensor<half, Shape<1,1,1,-1,D>, Stride<1,1,1,D,1>>;
    TileH s; TASSIGN(s, 0); s.SetValidRow(D); s.SetValidCol(D);
    GmH gs(const_cast<__gm__ half*>(S_gm), Shape<1,1,1,-1,D>(D));
    TLOAD(s, gs);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    GmH gh(final_gm, Shape<1,1,1,-1,D>(D));
    TSTORE(gh, s);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

// CfgWS = w@S (in_nt=2), CfgKV = k^T@vc (in_nt=3), P = scan params (B,H,NC,C,D).
template <typename CfgWS, typename CfgKV, typename P>
__global__ AICORE void scan_kernel(__gm__ const half* w, __gm__ const half* u,
                                   __gm__ const half* k, __gm__ const float* decay,
                                   __gm__ char* ws, __gm__ half* h_out, __gm__ half* final,
                                   uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);
    constexpr unsigned C = P::C, D = P::D, H = P::H, NC = P::NC;
    constexpr unsigned UNITS = P::B * P::H;
    using WM = WsMap<P>;
    unsigned cid = get_block_idx();
    unsigned bnum = get_block_num();

    __gm__ half*  S_base  = (__gm__ half*) (ws + (int64_t)cid * WM::BYTES + WM::S_OFF);
    __gm__ float* WS_base = (__gm__ float*)(ws + (int64_t)cid * WM::BYTES + WM::WS_OFF);
    __gm__ float* KV_base = (__gm__ float*)(ws + (int64_t)cid * WM::BYTES + WM::KV_OFF);
    __gm__ half*  VC_base = (__gm__ half*) (ws + (int64_t)cid * WM::BYTES + WM::VC_OFF);

    if constexpr (DAV_VEC) { set_mask_norm(); set_vector_mask((uint64_t)-1, (uint64_t)-1); }

    for (unsigned unit = cid; unit < UNITS; unit += bnum) {
        unsigned b = unit / H;
        unsigned h = unit % H;

        if constexpr (DAV_VEC) {
            vec_zero_state<P>(S_base);
            ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, F_REC));  // prime S ready
        }

        for (unsigned c = 0; c < NC; ++c) {
            int64_t tile_off = ((int64_t)(b * NC + c) * C * H + h) * D;   // base of [C,D] tile in w/u/k
            int64_t hout_off = ((int64_t)(b * NC + c) * H + h) * D * D;

            if constexpr (DAV_CUBE) {
                wait_flag_dev(F_REC); pipe_barrier(PIPE_ALL);
                pto_einsum::matmul_one_tile_deep<half, CfgWS, false, true>(w + tile_off, S_base, WS_base, 0);
                ffts_cross_core_sync(PIPE_FIX, pto_einsum::GetffstMsg(0x2, F_MM1));

                wait_flag_dev(F_RES); pipe_barrier(PIPE_ALL);
                pto_einsum::matmul_one_tile_deep<half, CfgKV, false, true>(k + tile_off, VC_base, KV_base, 0);
                ffts_cross_core_sync(PIPE_FIX, pto_einsum::GetffstMsg(0x2, F_MM2));
            }
            if constexpr (DAV_VEC) {
                wait_flag_dev(F_MM1); pipe_barrier(PIPE_ALL);
                vec_readout_residual<P>(u, WS_base, S_base, h_out + hout_off, VC_base, tile_off);
                ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, F_RES));

                wait_flag_dev(F_MM2); pipe_barrier(PIPE_ALL);
#ifdef SCAN_PERDIM_DECAY
                __gm__ const float* dvec = decay + (int64_t)((b * H + h) * NC + c) * D;
                vec_recurrence_perdim<P>(S_base, KV_base, dvec);
#else
                float dval = decay[(int64_t)(b * H + h) * NC + c];
                vec_recurrence<P>(S_base, KV_base, dval);
#endif
                ffts_cross_core_sync(PIPE_MTE3, pto_einsum::GetffstMsg(0x2, F_REC));
            }
        }

        if constexpr (DAV_CUBE) { wait_flag_dev(F_REC); pipe_barrier(PIPE_ALL); }  // drain last recur
        if constexpr (DAV_VEC) {
            int64_t fin_off = ((int64_t)(b * H + h)) * D * D;
            vec_store_final<P>(S_base, final + fin_off);
        }
    }
}

} // namespace chunk_h_scan
