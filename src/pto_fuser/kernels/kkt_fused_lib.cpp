// Config + host ABI for the fused-kkt prototype. Shape: kc_exp [NC, C, H, D],
// "bihd,bjhd->bihj" NT direct-read. NC (#chunks), H (#heads), C (chunk size) and
// D (head dim) all come from -DKKT_NC / -DKKT_H / -DKKT_C / -DKKT_D so one .so serves
// a given (nc,H,C,D); rebuilt per workload. C and D default to 128 (GDN/KDA), but the
// zoo mechanisms run smaller chunks/heads (e.g. C=16, D=64) — the matmul geom clamps
// its 128 tiles to min(tile, padded-dim), so the only C/D-dependence is the strides
// here and the score-dim (n_free0) the epilogue indexes by.
#include "kkt_fused.h"

#ifndef KKT_NC
#define KKT_NC 512
#endif
#ifndef KKT_H
#define KKT_H 32
#endif
#ifndef KKT_C
#define KKT_C 128
#endif
#ifndef KKT_D
#define KKT_D 128
#endif

#ifdef KKT_NATIVE
// Step 3: native [M,C,D] layout (M = nc*H, heads OUTER) — the Program's own batch.
// One batch axis, contiguous [M,C,D] operands → the matmul-core reads them with no
// transpose, and the epilogue indexes g/β/L by the flat M index (CFG::native path in
// kkt_epilogue_one). This is the plain `nid,njd->nij` batched-NT config; the gated/
// masked epilogue is bolted onto it with zero layout bridge (vs the mega config below).
struct config_einsum {
    static const unsigned n_free0 = KKT_C;     // i
    static const unsigned n_free1 = KKT_C;     // j
    static const unsigned n_contract = KKT_D;  // d
    static const unsigned n_inplace = KKT_NC * KKT_H;   // M batches
    static const int32_t kkt_H = KKT_H;
    static const int64_t kkt_T = (int64_t)KKT_NC * KKT_C;
    static constexpr bool native = true;

    static const unsigned tile_m = 128;
    static const unsigned tile_n = 128;
    static const unsigned tile_k = 128;

    static constexpr unsigned out_fusible = 0;
    static constexpr unsigned out_row_stride = 0;
    static constexpr unsigned out_n_batch = 0;
    static constexpr unsigned out_batch_sizes[1] = {0};
    static constexpr unsigned out_batch_strides[1] = {0};

    // NT, single M batch over contiguous [M,C,D]: row i stride = D, k innermost.
    static constexpr unsigned in_nt = 1;
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = KKT_D;
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = KKT_D;
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[1] = {KKT_NC * KKT_H};
    static constexpr unsigned in0_batch_strides[1] = {KKT_C * KKT_D};
    static constexpr unsigned in1_batch_strides[1] = {KKT_C * KKT_D};
};
#else
struct config_einsum {
    static const unsigned n_free0 = KKT_C;     // i
    static const unsigned n_free1 = KKT_C;     // j
    static const unsigned n_contract = KKT_D;  // d
    static const unsigned n_inplace = KKT_NC * KKT_H;
    static const int32_t kkt_H = KKT_H;
    static const int64_t kkt_T = (int64_t)KKT_NC * KKT_C;
    static constexpr bool native = false;

    static const unsigned tile_m = 128;
    static const unsigned tile_n = 128;
    static const unsigned tile_k = 128;

    // Output side unused (FUSE_OUT=false; the epilogue writes L). Neutral.
    static constexpr unsigned out_fusible = 0;
    static constexpr unsigned out_row_stride = 0;
    static constexpr unsigned out_n_batch = 0;
    static constexpr unsigned out_batch_sizes[1] = {0};
    static constexpr unsigned out_batch_strides[1] = {0};

    // NT strided-input: contraction d innermost+contiguous on both operands.
    static constexpr unsigned in_nt = 1;       // NT (B transposed via DN)
    static constexpr unsigned in_n_batch = 2;  // batch axes [nc, H]
    static constexpr unsigned in0_row_stride = KKT_H * KKT_D;  // free0 (i) src stride
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = KKT_H * KKT_D;  // free1 (j) src stride
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[2] = {KKT_NC, KKT_H};
    static constexpr unsigned in0_batch_strides[2] = {KKT_C * KKT_H * KKT_D, KKT_D};
    static constexpr unsigned in1_batch_strides[2] = {KKT_C * KKT_H * KKT_D, KKT_D};
};
#endif

using pto_einsum::cached_core_num;

extern "C" {

// Allocate ws_res once (I * C * C floats). NT reads k directly -> no input workspace.
void* kkt_setup() {
    size_t elems = (size_t)config_einsum::n_inplace * KKT_C * KKT_C;
    void* ws = nullptr;
    aclrtMalloc(&ws, elems * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void kkt_exec(const half* a, const half* b, const float* g_t, const half* beta_t,
              const float* mask, float* ws_res, half* L, int32_t Hh, int64_t Tt, void* stream) {
    (void)Hh; (void)Tt;  // now compile-time via config_einsum
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    kkt_fused::kkt_fused_kernel<config_einsum><<<cores, nullptr, stream>>>(
        a, b, g_t, beta_t, mask, ws_res, L, faddr);
}

void kkt_teardown(void* ws) { if (ws) aclrtFree(ws); }

// ── V2: per-tile interleave ─────────────────────────────────────────────────
// Tiny ring: block_num * 2 slots of C*C floats (not I*C*C). L2-resident.
void* kkt_setup_v2() {
    int64_t cores = cached_core_num();
    size_t elems = (size_t)cores * 2 * KKT_C * KKT_C;
    void* ws = nullptr;
    aclrtMalloc(&ws, elems * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void kkt_exec_v2(const half* a, const half* b, const float* g_t, const half* beta_t,
                 const float* mask, float* ws_ping, half* L, int32_t Hh, int64_t Tt, void* stream) {
    (void)Hh; (void)Tt;
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    kkt_fused::kkt_fused_kernel_v2<config_einsum><<<cores, nullptr, stream>>>(
        a, b, g_t, beta_t, mask, ws_ping, L, faddr);
}

}
