// Config + host ABI for the fused chunk_o "flash" prototype (qkv_fused.h). Two
// native configs keyed by -DQKV_NC/_H/_C/_D/_DV (one .so per (nc,H,C,d_k,d_v)):
//   config_qk : S = q@kᵀ, native NT [M,C,D]  (== kkt's KKT_NATIVE config, q,k operands)
//   config_av : o = A@v,  native NN [M,C,C]@[M,C,DV]  (plain contiguous, in_nt=0)
// M = nc*H (heads outer). C/D default 128 (GDN); the zoo runs C=16, d_k=d_v=64.
#include "qkv_fused.h"

#ifndef QKV_NC
#define QKV_NC 512
#endif
#ifndef QKV_H
#define QKV_H 32
#endif
#ifndef QKV_C
#define QKV_C 128
#endif
#ifndef QKV_D
#define QKV_D 128
#endif
#ifndef QKV_DV
#define QKV_DV QKV_D
#endif

// Stage 1: S = q @ kᵀ — native NT over contiguous [M,C,D], contraction d innermost
// on both operands (identical to kkt_fused_lib.cpp's KKT_NATIVE config_einsum).
struct config_qk {
    static const unsigned n_free0 = QKV_C;     // i
    static const unsigned n_free1 = QKV_C;     // j
    static const unsigned n_contract = QKV_D;  // d
    static const unsigned n_inplace = QKV_NC * QKV_H;
    static const int32_t kkt_H = QKV_H;
    static const int64_t kkt_T = (int64_t)QKV_NC * QKV_C;
    static constexpr bool native = true;

    static const unsigned tile_m = 128;
    static const unsigned tile_n = 128;
    static const unsigned tile_k = 128;

    static constexpr unsigned out_fusible = 0;
    static constexpr unsigned out_row_stride = 0;
    static constexpr unsigned out_n_batch = 0;
    static constexpr unsigned out_batch_sizes[1] = {0};
    static constexpr unsigned out_batch_strides[1] = {0};

    static constexpr unsigned in_nt = 1;       // NT (B transposed via DN)
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = QKV_D;
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = QKV_D;
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[1] = {QKV_NC * QKV_H};
    static constexpr unsigned in0_batch_strides[1] = {QKV_C * QKV_D};
    static constexpr unsigned in1_batch_strides[1] = {QKV_C * QKV_D};
};

// Stage 3: o = A @ v — native NN over contiguous [M,C,C] @ [M,C,DV]. in_nt=0 => the
// core's default contiguous read (A row-major [C,C] with K=C inner; v row-major
// [C,DV] with K=C the row stride, free1=DV inner). Strides below are ignored on the
// NT_IN=false path (A_ROW_STRIDE=K, A_K_STRIDE=1, B_K_STRIDE=Bcols, B_N_STRIDE=1);
// only the batch strides matter for the M batching.
struct config_av {
    static const unsigned n_free0 = QKV_C;      // i
    static const unsigned n_free1 = QKV_DV;     // e
    static const unsigned n_contract = QKV_C;   // j
    static const unsigned n_inplace = QKV_NC * QKV_H;

    static const unsigned tile_m = 128;
    static const unsigned tile_n = 128;
    static const unsigned tile_k = 128;

    static constexpr unsigned out_fusible = 0;
    static constexpr unsigned out_row_stride = 0;
    static constexpr unsigned out_n_batch = 0;
    static constexpr unsigned out_batch_sizes[1] = {0};
    static constexpr unsigned out_batch_strides[1] = {0};

    static constexpr unsigned in_nt = 0;        // NN contiguous (default read)
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = 0;
    static constexpr unsigned in0_k_stride = 0;
    static constexpr unsigned in1_col_stride = 0;
    static constexpr unsigned in1_k_stride = 0;
    static constexpr unsigned in_batch_sizes[1] = {QKV_NC * QKV_H};
    static constexpr unsigned in0_batch_strides[1] = {QKV_C * QKV_C};
    static constexpr unsigned in1_batch_strides[1] = {QKV_C * QKV_DV};
};

// Stage 3 (V2): single-batch NN [C,C]@[C,DV] — A read from the per-core slot (base
// offset 0), v[pid]/o[pid] passed as offset base pointers by the kernel.
struct config_av_tile {
    static const unsigned n_free0 = QKV_C;
    static const unsigned n_free1 = QKV_DV;
    static const unsigned n_contract = QKV_C;
    static const unsigned n_inplace = 1;

    static const unsigned tile_m = 128;
    static const unsigned tile_n = 128;
    static const unsigned tile_k = 128;

    static constexpr unsigned out_fusible = 0;
    static constexpr unsigned out_row_stride = 0;
    static constexpr unsigned out_n_batch = 0;
    static constexpr unsigned out_batch_sizes[1] = {0};
    static constexpr unsigned out_batch_strides[1] = {0};

    static constexpr unsigned in_nt = 0;
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = 0;
    static constexpr unsigned in0_k_stride = 0;
    static constexpr unsigned in1_col_stride = 0;
    static constexpr unsigned in1_k_stride = 0;
    static constexpr unsigned in_batch_sizes[1] = {1};
    static constexpr unsigned in0_batch_strides[1] = {0};
    static constexpr unsigned in1_batch_strides[1] = {0};
};

using pto_einsum::cached_core_num;

extern "C" {

// ws_qk [M,C,C] float (the S scratch) + ws_A [M,C,C] half (the masked-score scratch),
// packed in one allocation: [ ws_qk (M*C*C f32) | ws_A (M*C*C f16) ].
void* qkv_setup() {
    size_t M = (size_t)config_qk::n_inplace;
    size_t bytes = M * QKV_C * QKV_C * sizeof(float) + M * QKV_C * QKV_C * sizeof(half);
    void* ws = nullptr;
    aclrtMalloc(&ws, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void qkv_exec(const half* q, const half* k, const half* v, const float* g_t,
              const half* beta_t, const float* mask, void* ws, float* o, void* stream) {
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    size_t M = (size_t)config_qk::n_inplace;
    float* ws_qk = reinterpret_cast<float*>(ws);
    half*  ws_A  = reinterpret_cast<half*>(ws_qk + M * QKV_C * QKV_C);
    qkv_fused::qkv_flash_kernel_v1<config_qk, config_av><<<cores, nullptr, stream>>>(
        q, k, v, g_t, beta_t, mask, ws_qk, ws_A, o, faddr);
}

void qkv_teardown(void* ws) { if (ws) aclrtFree(ws); }

// ── V2: per-tile interleave, double-buffered. Tiny per-core L2 ring: [cores]×2
// slots of C*C float (S) + C*C half (A) — never an [M,C,C] HBM materialization.
void* qkv_setup_v2() {
    int64_t cores = cached_core_num();
    size_t bytes = (size_t)cores * 2 * QKV_C * QKV_C * sizeof(float)
                 + (size_t)cores * 2 * QKV_C * QKV_C * sizeof(half);
    void* ws = nullptr;
    aclrtMalloc(&ws, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void qkv_exec_v2(const half* q, const half* k, const half* v, const float* g_t,
                 const half* beta_t, const float* mask, void* ws, float* o, void* stream) {
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    float* s_ring = reinterpret_cast<float*>(ws);
    half*  a_ring = reinterpret_cast<half*>(s_ring + (size_t)cores * 2 * QKV_C * QKV_C);
    qkv_fused::qkv_flash_kernel_v2<config_qk, config_av_tile><<<cores, nullptr, stream>>>(
        q, k, v, g_t, beta_t, mask, s_ring, a_ring, o, faddr);
}

}
