// Config + host ABI for the per-dim PROLOGUE prototype (prologue_fused.h). Native
// [M,C,D] layout (M = nc*H, heads outer) — the same plain `nid,njd->nij` NT config the
// kkt native path uses, here driving q̃@k̂ᵀ with the decay pre-applied to the operands
// (Vec prologue) instead of a scalar epilogue. C (chunk), D (head dim), NC, H from -D.
#include "prologue_fused.h"

#ifndef PRO_NC
#define PRO_NC 4
#endif
#ifndef PRO_H
#define PRO_H 8
#endif
#ifndef PRO_C
#define PRO_C 128
#endif
#ifndef PRO_D
#define PRO_D 128
#endif

struct config_einsum {
    static const unsigned n_free0 = PRO_C;     // i (chunk row)
    static const unsigned n_free1 = PRO_C;     // j (chunk col)
    static const unsigned n_contract = PRO_D;  // d (head dim)
    static const unsigned n_inplace = PRO_NC * PRO_H;   // M batches
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
    static constexpr unsigned in0_row_stride = PRO_D;
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = PRO_D;
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[1] = {PRO_NC * PRO_H};
    static constexpr unsigned in0_batch_strides[1] = {PRO_C * PRO_D};
    static constexpr unsigned in1_batch_strides[1] = {PRO_C * PRO_D};
};

// Single-tile config for the V2 per-tile matmul: ONE [C,D]@[C,D]ᵀ batch read from a
// ring slot (base offset 0). Same NT strides as config_einsum but n_inplace=1, so the
// matmul's batch decode gives baseA=baseB=0 -> it reads opA/opB directly.
struct config_tile {
    static const unsigned n_free0 = PRO_C;
    static const unsigned n_free1 = PRO_C;
    static const unsigned n_contract = PRO_D;
    static const unsigned n_inplace = 1;
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
    static constexpr unsigned in0_row_stride = PRO_D;
    static constexpr unsigned in0_k_stride = 1;
    static constexpr unsigned in1_col_stride = PRO_D;
    static constexpr unsigned in1_k_stride = 1;
    static constexpr unsigned in_batch_sizes[1] = {1};
    static constexpr unsigned in0_batch_strides[1] = {PRO_C * PRO_D};
    static constexpr unsigned in1_batch_strides[1] = {PRO_C * PRO_D};
};

using pto_einsum::cached_core_num;

// One scratch block holds qd_s [M,C,D] half | kinv_s [M,C,D] half | ws_res [M,C,C] f32.
static const size_t QD_BYTES   = (size_t)config_einsum::n_inplace * PRO_C * PRO_D * sizeof(half);
static const size_t KINV_BYTES = QD_BYTES;
static const size_t WS_BYTES    = (size_t)config_einsum::n_inplace * PRO_C * PRO_C * sizeof(float);

extern "C" {

void* pro_setup() {
    void* ws = nullptr;
    aclrtMalloc(&ws, QD_BYTES + KINV_BYTES + WS_BYTES, ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void pro_exec(const half* q, const half* P, const half* k, const half* invP,
              const float* mask, void* ws, half* L, int32_t unused0, int64_t unused1,
              void* stream) {
    (void)unused0; (void)unused1;
    char* base = (char*)ws;
    half*  qd_s   = (half*)(base);
    half*  kinv_s = (half*)(base + QD_BYTES);
    float* ws_res = (float*)(base + QD_BYTES + KINV_BYTES);
    int64_t cores = cached_core_num();
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    prologue_fused::qk_prologue_kernel<config_einsum><<<cores, nullptr, stream>>>(
        q, P, k, invP, mask, qd_s, kinv_s, ws_res, L, faddr);
}

void pro_teardown(void* ws) { if (ws) aclrtFree(ws); }

// ── V2: per-core L2-resident ring (no full [M,C,D] scratch) ──────────────────
// op_ring: cores * 2 * C*D halfs (opA|opB per core) ; a_ring: cores * C*C floats.
void* pro_setup_v2() {
    int64_t cores = cached_core_num();
    size_t op_bytes = (size_t)cores * 2 * PRO_C * PRO_D * sizeof(half);
    size_t a_bytes  = (size_t)cores * PRO_C * PRO_C * sizeof(float);
    void* ws = nullptr;
    aclrtMalloc(&ws, op_bytes + a_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    return ws;
}

void pro_exec_v2(const half* q, const half* P, const half* k, const half* invP,
                 const float* mask, void* ws, half* L, int32_t unused0, int64_t unused1,
                 void* stream) {
    (void)unused0; (void)unused1;
    int64_t cores = cached_core_num();
    size_t op_bytes = (size_t)cores * 2 * PRO_C * PRO_D * sizeof(half);
    char* base = (char*)ws;
    half*  op_ring = (half*)(base);
    float* a_ring  = (float*)(base + op_bytes);
    uint32_t flen = 0; uint64_t faddr = 0;
    rtGetC2cCtrlAddr(&faddr, &flen);
    prologue_fused::qk_prologue_kernel_v2<config_einsum, config_tile><<<cores, nullptr, stream>>>(
        q, P, k, invP, mask, op_ring, a_ring, L, faddr);
}

}
