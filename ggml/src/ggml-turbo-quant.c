/*
 * TurboQuant: KV cache compression via FWHT rotation + Lloyd-Max codebook
 * Based on: arXiv 2504.19874 (ICLR 2026)
 * Port to ik_llama.cpp from spiritbuun/buun-llama-cpp
 *
 * Key difference from original proposal: uses Fast Walsh-Hadamard Transform
 * (O(n log n)) instead of dense rotation matrix (O(n^2)), solving the
 * throughput regression that got the original rejected from ik_llama.cpp.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>

/* ---------- FWHT sign arrays (deterministic, seed=42) ---------- */
/* These must match the CUDA __constant__ arrays exactly */

static const float TURBO_WHT_S1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};
static const float TURBO_WHT_S2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

/* ---------- Lloyd-Max codebooks (optimal for post-FWHT Gaussian) ---------- */

static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static const float MIDPOINTS_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

static const float CENTROIDS_4BIT[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static const float MIDPOINTS_4BIT[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

/* ---------- FWHT: O(n log n) rotation ---------- */

static void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

static void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_S1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_S2[i];
}

static void turbo_rotate_inverse(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_S2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_S1[i];
}

/* ---------- nearest centroid lookups ---------- */

static int nearest_3bit(float val) {
    if      (val < MIDPOINTS_3BIT[0]) return 0;
    else if (val < MIDPOINTS_3BIT[1]) return 1;
    else if (val < MIDPOINTS_3BIT[2]) return 2;
    else if (val < MIDPOINTS_3BIT[3]) return 3;
    else if (val < MIDPOINTS_3BIT[4]) return 4;
    else if (val < MIDPOINTS_3BIT[5]) return 5;
    else if (val < MIDPOINTS_3BIT[6]) return 6;
    else                              return 7;
}

static int nearest_4bit(float val) {
    if (val < MIDPOINTS_4BIT[7]) {
        if (val < MIDPOINTS_4BIT[3]) {
            if (val < MIDPOINTS_4BIT[1]) return val < MIDPOINTS_4BIT[0] ? 0 : 1;
            else                         return val < MIDPOINTS_4BIT[2] ? 2 : 3;
        } else {
            if (val < MIDPOINTS_4BIT[5]) return val < MIDPOINTS_4BIT[4] ? 4 : 5;
            else                         return val < MIDPOINTS_4BIT[6] ? 6 : 7;
        }
    } else {
        if (val < MIDPOINTS_4BIT[11]) {
            if (val < MIDPOINTS_4BIT[9])  return val < MIDPOINTS_4BIT[8] ? 8 : 9;
            else                          return val < MIDPOINTS_4BIT[10] ? 10 : 11;
        } else {
            if (val < MIDPOINTS_4BIT[13]) return val < MIDPOINTS_4BIT[12] ? 12 : 13;
            else                          return val < MIDPOINTS_4BIT[14] ? 14 : 15;
        }
    }
}

/* ========== TURBO3_0: 3-bit (2-bit low + 1-bit sign) ========== */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_TURBO3_GROUP == 0);
    const int ngroups = k / QK_TURBO3_GROUP;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;

    for (int g = 0; g < ngroups; g++) {
        const float * src = x + g * QK_TURBO3_GROUP;

        /* Step 1: compute group norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        /* Step 2: normalize and rotate */
        float rot[128];
        for (int j = 0; j < 128; j++) rot[j] = src[j] * inv_norm;
        turbo_rotate_forward(rot);

        /* Step 3: quantize to 3-bit and compute reconstruction norm */
        float recon_norm_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &y[g * blocks_per_group + b];
            const int off = b * QK_TURBO3;
            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_3bit(rot[off + j]);
                blk->qs[j / 4] |= (uint8_t)((idx & 0x3) << ((j % 4) * 2));
                if (idx & 0x4) blk->signs[j / 8] |= (uint8_t)(1 << (j % 8));
                recon_norm_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        /* Step 4: norm correction */
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            y[g * blocks_per_group + b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void quantize_row_turbo3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo3_0_ref(x, (block_turbo3_0 *)y, k);
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_TURBO3_GROUP == 0);
    const int ngroups = k / QK_TURBO3_GROUP;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;

    for (int g = 0; g < ngroups; g++) {
        float rot[128];

        /* Unpack indices and look up centroids */
        for (int b = 0; b < blocks_per_group; b++) {
            const block_turbo3_0 * blk = &x[g * blocks_per_group + b];
            float norm = GGML_FP16_TO_FP32(blk->norm);
            const int off = b * QK_TURBO3;

            for (int j = 0; j < QK_TURBO3; j++) {
                uint8_t low2 = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                uint8_t hi1  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
                uint8_t idx  = low2 | (hi1 << 2);
                rot[off + j] = CENTROIDS_3BIT[idx] * norm;
            }
        }

        /* Inverse rotate */
        turbo_rotate_inverse(rot);

        /* Write output */
        memcpy(y + g * QK_TURBO3_GROUP, rot, 128 * sizeof(float));
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    GGML_ASSERT(n_per_row % QK_TURBO3_GROUP == 0);
    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}

/* ========== TURBO4_0: 4-bit PolarQuant ========== */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_TURBO4;

        /* Step 1: compute norm */
        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
        float norm = sqrtf(norm_sq);
        float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

        /* Step 2: normalize and rotate */
        float rot[128];
        for (int j = 0; j < 128; j++) rot[j] = src[j] * inv_norm;
        turbo_rotate_forward(rot);

        /* Step 3: 4-bit quantize */
        for (int j = 0; j < 128; j += 2) {
            uint8_t idx0 = (uint8_t)nearest_4bit(rot[j]);
            uint8_t idx1 = (uint8_t)nearest_4bit(rot[j + 1]);
            y[block].qs[j / 2] = (idx1 << 4) | idx0;
        }

        /* Step 4: norm correction */
        float recon_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            uint8_t idx = (j & 1) ? (y[block].qs[j / 2] >> 4) : (y[block].qs[j / 2] & 0xF);
            recon_sq += CENTROIDS_4BIT[idx] * CENTROIDS_4BIT[idx];
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);
    }
}

void quantize_row_turbo4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_turbo4_0_ref(x, (block_turbo4_0 *)y, k);
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 4-bit indices and reconstruct in rotated space */
        float rot[128];
        for (int j = 0; j < 128; j++) {
            uint8_t idx = (j & 1) ? (x[block].qs[j / 2] >> 4) : (x[block].qs[j / 2] & 0xF);
            rot[j] = CENTROIDS_4BIT[idx] * norm;
        }

        /* Inverse rotate and write output */
        turbo_rotate_inverse(rot);
        memcpy(y + block * QK_TURBO4, rot, 128 * sizeof(float));
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    GGML_ASSERT(n_per_row % QK_TURBO4 == 0);
    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row);
    }
    return nrows * row_size;
}

/* ---------- FWHT op for ggml graph (CPU backend) ---------- */

void ggml_compute_turbo_wht(struct ggml_tensor * dst, int ith, int nth) {
    const struct ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t ne0 = src->ne[0];
    GGML_ASSERT(ne0 % 128 == 0);
    const int64_t ngroups_per_row = ne0 / 128;
    const int64_t nrows = (int64_t)src->ne[1] * src->ne[2] * src->ne[3];
    const int64_t total_groups = nrows * ngroups_per_row;

    const int64_t groups_per_thread = (total_groups + nth - 1) / nth;
    const int64_t first = groups_per_thread * ith;
    const int64_t last  = first + groups_per_thread < total_groups ? first + groups_per_thread : total_groups;

    for (int64_t g = first; g < last; g++) {
        const int64_t row = g / ngroups_per_row;
        const int64_t col = (g % ngroups_per_row) * 128;

        const float * s = (const float *)((const char *)src->data + row * src->nb[1]) + col;
        float * d = (float *)((char *)dst->data + row * dst->nb[1]) + col;

        float buf[128];
        memcpy(buf, s, 128 * sizeof(float));

        if (direction == 0) {
            turbo_rotate_forward(buf);
        } else {
            turbo_rotate_inverse(buf);
        }

        memcpy(d, buf, 128 * sizeof(float));
    }
}
