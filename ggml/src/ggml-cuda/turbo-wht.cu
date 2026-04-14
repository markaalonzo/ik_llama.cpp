#include "turbo-wht.cuh"
#include "../ggml-turbo-quant-data.h"

// Sign arrays for FWHT rotation (deterministic, seed=42)
// Must match the CPU arrays in ggml-turbo-quant.c exactly.
static __constant__ float d_turbo_wht_s1[] = TURBO_WHT_S1_INIT;
static __constant__ float d_turbo_wht_s2[] = TURBO_WHT_S2_INIT;

// One CUDA block per 128-element group. 128 threads per block.
// 64 active threads for butterfly, all 128 for load/store.
static __global__ void k_turbo_wht(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements, const int direction) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    const float * s_first  = (direction == 0) ? d_turbo_wht_s1 : d_turbo_wht_s2;
    const float * s_second = (direction == 0) ? d_turbo_wht_s2 : d_turbo_wht_s1;

    __shared__ float buf[128];

    // Kernel is launched with exactly 128 threads per block (see launch
    // config below). All threads participate in load/store; the butterfly
    // only uses the first 64 threads (each processes one pair).

    // Load and apply first sign flip
    buf[threadIdx.x] = src[offset + threadIdx.x] * s_first[threadIdx.x];
    __syncthreads();

    // Parallel FWHT butterfly: 64 threads, 7 passes (log2(128) = 7)
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b;
            buf[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second sign flip, write output
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128 * s_second[threadIdx.x];
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(src0);
    const int64_t n_groups = n_elements / 128;

    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_elements, direction);
}
