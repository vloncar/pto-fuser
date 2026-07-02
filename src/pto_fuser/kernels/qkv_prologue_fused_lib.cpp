// Config + host ABI for the per-dim flash prototype (qkv_prologue_fused.h). Same two
// native configs as qkv_fused_lib.cpp — config_qk (NT, for the prescale + score) and
// config_av (NN, for A·v) — keyed by -DQKV_NC/_H/_C/_D/_DV (one .so per shape).
#include "qkv_prologue_fused.h"

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

// S = q̃ @ k̂ᵀ (and the prescale) — native NT over contiguous [M,C,D].
struct config_qk {
    static const unsigned n_free0 = QKV_C;
    static const unsigned n_free1 = QKV_C;
    static const unsigned n_contract = QKV_D;
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
    static constexpr unsigned in_nt = 1;
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = QKV_D;
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = QKV_D;
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[1] = {QKV_NC * QKV_H};
    static constexpr unsigned in0_batch_strides[1] = {QKV_C * QKV_D};
    static constexpr unsigned in1_batch_strides[1] = {QKV_C * QKV_D};
};

// o = A @ v — native NN over contiguous [M,C,C] @ [M,C,DV].
struct config_av {
    static const unsigned n_free0 = QKV_C;
    static const unsigned n_free1 = QKV_DV;
    static const unsigned n_contract = QKV_C;
    static const unsigned n_inplace = QKV_NC * QKV_H;
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
    static constexpr unsigned in_batch_sizes[1] = {QKV_NC * QKV_H};
    static constexpr unsigned in0_batch_strides[1] = {QKV_C * QKV_C};
    static constexpr unsigned in1_batch_strides[1] = {QKV_C * QKV_DV};
};

// Single-batch NT [C,D]@[C,D]ᵀ — the V2 score reads q̃/k̂ from a per-core op slot.
struct config_qk_tile {
    static const unsigned n_free0 = QKV_C;
    static const unsigned n_free1 = QKV_C;
    static const unsigned n_contract = QKV_D;
    static const unsigned n_inplace = 1;
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
    static constexpr unsigned in_nt = 1;
    static constexpr unsigned in_n_batch = 1;
    static constexpr unsigned in0_row_stride = QKV_D;
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = QKV_D;
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[1] = {1};
    static constexpr unsigned in0_batch_strides[1] = {0};
    static constexpr unsigned in1_batch_strides[1] = {0};
};

// Single-batch NN [C,C]@[C,DV] — the V2 output reads A from the slot, v[pid] as base.
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

// Scratch: qd_s + kinv_s ([M,C,D] half each) | ws_qk ([M,C,C] f32) | ws_A ([M,C,C] half).
void* qkvp_setup() {
    size_t M = (size_t)config_qk::n_inplace;
    size_t bytes = 2 * M * QKV_C * QKV_D * sizeof(half)
                 + M * QKV_C * QKV_C * sizeof(float)
                 + M * QKV_C * QKV_C * sizeof(half);
    void* ws = nullptr;
    aclrtMalloc(&ws, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void qkvp_exec(const half* q, const half* coef_ag, const half* k, const half* coef_bg,
               const float* mask, const half* v, void* ws, float* o, void* stream) {
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    size_t M = (size_t)config_qk::n_inplace;
    half*  qd_s  = reinterpret_cast<half*>(ws);
    half*  kinv_s = qd_s + M * QKV_C * QKV_D;
    float* ws_qk = reinterpret_cast<float*>(kinv_s + M * QKV_C * QKV_D);
    half*  ws_A  = reinterpret_cast<half*>(ws_qk + M * QKV_C * QKV_C);
    qkv_prologue_fused::qkv_prologue_flash_v1<config_qk, config_av><<<cores, nullptr, stream>>>(
        q, coef_ag, k, coef_bg, mask, qd_s, kinv_s, ws_qk, ws_A, v, o, faddr);
}

void qkvp_teardown(void* ws) { if (ws) aclrtFree(ws); }

// ── V2: double-buffered interleave. Per-core L2 ring: 2×(qd|kinv) half + 2×S f32
// + 2×A half — ops, S and A never hit HBM.
void* qkvp_setup_v2() {
    int64_t cores = cached_core_num();
    size_t bytes = (size_t)cores * 2 * (2 * QKV_C * QKV_D) * sizeof(half)   // op ring
                 + (size_t)cores * 2 * QKV_C * QKV_C * sizeof(float)        // S ring
                 + (size_t)cores * 2 * QKV_C * QKV_C * sizeof(half);        // A ring
    void* ws = nullptr;
    aclrtMalloc(&ws, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void qkvp_exec_v2(const half* q, const half* coef_ag, const half* k, const half* coef_bg,
                  const float* mask, const half* v, void* ws, float* o, void* stream) {
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    half*  op_ring = reinterpret_cast<half*>(ws);
    float* s_ring = reinterpret_cast<float*>(op_ring + (size_t)cores * 2 * (2 * QKV_C * QKV_D));
    half*  a_ring = reinterpret_cast<half*>(s_ring + (size_t)cores * 2 * QKV_C * QKV_C);
    qkv_prologue_fused::qkv_prologue_flash_v2<config_qk, config_qk_tile, config_av_tile>
        <<<cores, nullptr, stream>>>(q, coef_ag, k, coef_bg, mask, op_ring, s_ring, a_ring,
                                     v, o, faddr);
}

}
