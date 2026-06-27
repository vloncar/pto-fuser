// Config + host ABI for the T2 cross-chunk-scan prototype.
// Operands token-major [B, NC, C, H, D] half; D=C=128. Two matmul configs feed
// matmul_one_tile_deep (NT_IN=true): CfgWS (w@S, in_nt=2) and CfgKV (k^T@vc,
// in_nt=3). B/H/NC come from -DSCAN_B / -DSCAN_H / -DSCAN_NC.
#include "scan_fused.h"

#ifndef SCAN_B
#define SCAN_B 1
#endif
#ifndef SCAN_H
#define SCAN_H 2
#endif
#ifndef SCAN_NC
#define SCAN_NC 3
#endif
#define SCAN_C 128
#define SCAN_D 128

struct scan_params {
    static constexpr unsigned B = SCAN_B;
    static constexpr unsigned H = SCAN_H;
    static constexpr unsigned NC = SCAN_NC;
    static constexpr unsigned C = SCAN_C;
    static constexpr unsigned D = SCAN_D;
};

// Common output-side fields (unused: FUSE_OUT=false, plain ws_res store).
#define SCAN_OUT_NEUTRAL \
    static constexpr unsigned out_fusible = 0; \
    static constexpr unsigned out_row_stride = 0; \
    static constexpr unsigned out_n_batch = 0; \
    static constexpr unsigned out_batch_sizes[1] = {0}; \
    static constexpr unsigned out_batch_strides[1] = {0};

// w @ S : M=C (w rows), N=D (S cols), K=D (contraction). in_nt=2 (NN-strided):
//   A=w natural ND  (row stride H*D over tokens, contraction d innermost, stride 1)
//   B=S strided-K   (col stride 1, contraction = S row, stride D)
struct CfgWS {
    static const unsigned n_free0 = SCAN_C;
    static const unsigned n_free1 = SCAN_D;
    static const unsigned n_contract = SCAN_D;
    static const unsigned n_inplace = 1;
    static const unsigned tile_m = 128, tile_n = 128, tile_k = 128;
    SCAN_OUT_NEUTRAL
    static constexpr unsigned in_nt = 2;
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = SCAN_H * SCAN_D;  // w token (free0) stride
    static constexpr unsigned in0_k_stride = 1;                  // w d (contraction) inner
    static constexpr unsigned in1_col_stride = 1;                // S col (free1)
    static constexpr unsigned in1_k_stride = SCAN_D;             // S row (contraction)
    static constexpr unsigned in_batch_sizes[1] = {1};
    static constexpr unsigned in0_batch_strides[1] = {0};
    static constexpr unsigned in1_batch_strides[1] = {0};
};

// k^T @ vc : M=D (k's d), N=D (vc's d), K=C (contraction over tokens). in_nt=3 (TN):
//   A=k transposed (free0 d stride 1, contraction = token, stride H*D)
//   B=vc NN-strided (col d stride 1, contraction = token, stride D; vc is contiguous [C,D])
struct CfgKV {
    static const unsigned n_free0 = SCAN_D;
    static const unsigned n_free1 = SCAN_D;
    static const unsigned n_contract = SCAN_C;
    static const unsigned n_inplace = 1;
    static const unsigned tile_m = 128, tile_n = 128, tile_k = 128;
    SCAN_OUT_NEUTRAL
    static constexpr unsigned in_nt = 3;
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = 1;               // k d (free0) contiguous (TN)
    static constexpr unsigned in0_k_stride = SCAN_H * SCAN_D;    // k token (contraction) stride
    static constexpr unsigned in1_col_stride = 1;               // vc d (free1)
    static constexpr unsigned in1_k_stride = SCAN_D;            // vc token (contraction) stride
    static constexpr unsigned in_batch_sizes[1] = {1};
    static constexpr unsigned in0_batch_strides[1] = {0};
    static constexpr unsigned in1_batch_strides[1] = {0};
};

using pto_einsum::cached_core_num;

extern "C" {

void* scan_setup() {
    int64_t cores = cached_core_num();
    size_t bytes = (size_t)cores * chunk_h_scan::WsMap<scan_params>::BYTES;
    void* ws = nullptr;
    aclrtMalloc(&ws, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemset(ws, bytes, 0, bytes);   // zero so the first vec_zero_state load is 0
    return ws;
}

void scan_exec(const half* w, const half* u, const half* k, const float* decay,
               void* ws, half* h_out, half* final, void* stream) {
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    chunk_h_scan::scan_kernel<CfgWS, CfgKV, scan_params><<<cores, nullptr, stream>>>(
        w, u, k, decay, (char*)ws, h_out, final, faddr);
}

void scan_teardown(void* ws) { if (ws) aclrtFree(ws); }

}
