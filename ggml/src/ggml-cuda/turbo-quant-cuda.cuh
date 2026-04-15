#pragma once
// Used by set-rows.cu to quantize rows into turbo3_0 / turbo4_0 blocks.
#include <cuda_runtime.h>
#include "../ggml-turbo-quant-data.h"
#include <cuda_fp16.h>
#include "ggml-common.h"

// Lloyd-Max codebooks in constant memory for fast lookup
static __constant__ float d_turbo_centroids_3bit[] = TURBO_CENTROIDS_3BIT_INIT;
static __constant__ float d_turbo_mid_3bit[] = TURBO_MIDPOINTS_3BIT_INIT;

static __constant__ float d_turbo_centroids_4bit[] = TURBO_CENTROIDS_4BIT_INIT;
static __constant__ float d_turbo_mid_4bit[] = TURBO_MIDPOINTS_4BIT_INIT;

static __constant__ float d_tq_wht_s1[] = TURBO_WHT_S1_INIT;
static __constant__ float d_tq_wht_s2[] = TURBO_WHT_S2_INIT;

// === Inline FWHT for set-rows kernels (single-thread, sequential) ===
static __device__ __forceinline__
void turbo_fwht_128_cuda(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

static __device__ __forceinline__
void turbo_rotate_forward_cuda(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= d_tq_wht_s1[i];
    turbo_fwht_128_cuda(x);
    for (int i = 0; i < 128; i++) x[i] *= d_tq_wht_s2[i];
}

// === Nearest centroid lookups ===
static __device__ __forceinline__
uint8_t turbo_find_nearest_3bit(float val) {
    if      (val < d_turbo_mid_3bit[0]) return 0;
    else if (val < d_turbo_mid_3bit[1]) return 1;
    else if (val < d_turbo_mid_3bit[2]) return 2;
    else if (val < d_turbo_mid_3bit[3]) return 3;
    else if (val < d_turbo_mid_3bit[4]) return 4;
    else if (val < d_turbo_mid_3bit[5]) return 5;
    else if (val < d_turbo_mid_3bit[6]) return 6;
    else                                return 7;
}

static __device__ __forceinline__
uint8_t turbo_find_nearest_4bit(float val) {
    if (val < d_turbo_mid_4bit[7]) {
        if (val < d_turbo_mid_4bit[3]) {
            if (val < d_turbo_mid_4bit[1]) return val < d_turbo_mid_4bit[0] ? 0 : 1;
            else                           return val < d_turbo_mid_4bit[2] ? 2 : 3;
        } else {
            if (val < d_turbo_mid_4bit[5]) return val < d_turbo_mid_4bit[4] ? 4 : 5;
            else                           return val < d_turbo_mid_4bit[6] ? 6 : 7;
        }
    } else {
        if (val < d_turbo_mid_4bit[11]) {
            if (val < d_turbo_mid_4bit[9])  return val < d_turbo_mid_4bit[8] ? 8 : 9;
            else                            return val < d_turbo_mid_4bit[10] ? 10 : 11;
        } else {
            if (val < d_turbo_mid_4bit[13]) return val < d_turbo_mid_4bit[12] ? 12 : 13;
            else                            return val < d_turbo_mid_4bit[14] ? 14 : 15;
        }
    }
}

// === TURBO4: inline quantize for set-rows ===
static __device__ __forceinline__
void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward_cuda(x);

    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j += 2) {
        uint8_t idx0 = turbo_find_nearest_4bit(x[j]);
        uint8_t idx1 = turbo_find_nearest_4bit(x[j + 1]);
        dst->qs[j / 2] = (idx1 << 4) | idx0;
        recon_sq += d_turbo_centroids_4bit[idx0] * d_turbo_centroids_4bit[idx0]
                  + d_turbo_centroids_4bit[idx1] * d_turbo_centroids_4bit[idx1];
    }
    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half((recon_norm > 1e-10f) ? norm / recon_norm : norm);
}

// === TURBO3: inline quantize for set-rows ===
static __device__ __forceinline__
void quantize_f32_turbo3_0_group(const float * src, block_turbo3_0 * dst_blocks) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward_cuda(x);

    float recon_norm_sq = 0.0f;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 * blk = &dst_blocks[b];
        const int off = b * QK_TURBO3;
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk->qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk->signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
            blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk->signs[j / 8] |= (1 << (j % 8));
            recon_norm_sq += d_turbo_centroids_3bit[idx] * d_turbo_centroids_3bit[idx];
        }
    }

    float recon_norm = sqrtf(recon_norm_sq);
    float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_blocks[b].norm = __float2half(corrected);
    }
}

// === Dequantize functions for flash attention ===

// TURBO3: dequantize one block_turbo3_0 (32 values) to float
static __device__ __forceinline__
void dequantize_turbo3_0_block(const block_turbo3_0 * blk, float * out) {
    float norm = __half2float(blk->norm);
    for (int j = 0; j < QK_TURBO3; j++) {
        uint8_t low2 = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        uint8_t hi1  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
        uint8_t idx  = low2 | (hi1 << 2);
        out[j] = d_turbo_centroids_3bit[idx] * norm;
    }
}

// TURBO4: dequantize 2 values (for fattn-vec float2 interface)
// QR_TURBO4_0 = 2: each call returns 2 values spaced 64 apart
#define QR_TURBO4_0 2
static __device__ __forceinline__
void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    {
        const int j = iqs;
        uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
        v.x = d_turbo_centroids_4bit[idx] * norm;
    }
    {
        const int j = iqs + 64;
        uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
        v.y = d_turbo_centroids_4bit[idx] * norm;
    }
}

// TURBO3: dequantize 2 values (for fattn-vec float2 interface)
// QR_TURBO3_0 = 2: each call returns 2 values spaced 16 apart
#define QR_TURBO3_0 2
static __device__ __forceinline__
void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    {
        const int j = iqs;
        const uint8_t low2 = (x[ib].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        const uint8_t hi1  = (x[ib].signs[j / 8] >> (j % 8)) & 0x1;
        v.x = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm;
    }
    {
        const int j = iqs + 16;
        const uint8_t low2 = (x[ib].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        const uint8_t hi1  = (x[ib].signs[j / 8] >> (j % 8)) & 0x1;
        v.y = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm;
    }
}
