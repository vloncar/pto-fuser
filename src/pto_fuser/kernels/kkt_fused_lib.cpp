// Config + host ABI for the fused-kkt prototype. Shape: kc_exp [NC, C, H, D],
// C=D=128, "bihd,bjhd->bihj" NT direct-read. NC (#chunks) and H (#heads) come from
// -DKKT_NC / -DKKT_H so one .so serves a given (nc,H); rebuilt per workload.
#include "kkt_fused.h"

#ifndef KKT_NC
#define KKT_NC 512
#endif
#ifndef KKT_H
#define KKT_H 32
#endif
#define KKT_C 128
#define KKT_D 128

struct config_einsum {
    static const unsigned n_free0 = KKT_C;     // i
    static const unsigned n_free1 = KKT_C;     // j
    static const unsigned n_contract = KKT_D;  // d
    static const unsigned n_inplace = KKT_NC * KKT_H;
    static const int32_t kkt_H = KKT_H;
    static const int64_t kkt_T = (int64_t)KKT_NC * KKT_C;

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

using pto_einsum::cached_core_num;

extern "C" {

// Allocate ws_res once (I * C * C floats). NT reads k directly -> no input workspace.
void* kkt_setup() {
    size_t elems = (size_t)config_einsum::n_inplace * KKT_C * KKT_C;
    void* ws = nullptr;
    aclrtMalloc(&ws, elems * sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void kkt_exec(const half* k, const float* g_t, const half* beta_t, const float* mask,
              float* ws_res, half* L, int32_t Hh, int64_t Tt, void* stream) {
    (void)Hh; (void)Tt;  // now compile-time via config_einsum
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    kkt_fused::kkt_fused_kernel<config_einsum><<<cores, nullptr, stream>>>(
        k, g_t, beta_t, mask, ws_res, L, faddr);
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

void kkt_exec_v2(const half* k, const float* g_t, const half* beta_t, const float* mask,
                 float* ws_ping, half* L, int32_t Hh, int64_t Tt, void* stream) {
    (void)Hh; (void)Tt;
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    kkt_fused::kkt_fused_kernel_v2<config_einsum><<<cores, nullptr, stream>>>(
        k, g_t, beta_t, mask, ws_ping, L, faddr);
}

}
