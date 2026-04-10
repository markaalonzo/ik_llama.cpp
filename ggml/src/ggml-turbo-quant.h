#pragma once

#include "ggml.h"
#include "ggml-common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// Dequantization
void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// High-level quantize (for type traits)
size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix);

// FWHT compute op (CPU backend)
void ggml_compute_turbo_wht(struct ggml_tensor * dst, int ith, int nth);

#ifdef __cplusplus
}
#endif
