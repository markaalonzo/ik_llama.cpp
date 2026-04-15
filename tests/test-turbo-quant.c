/*
 * TurboQuant round-trip test: quantize → dequantize → measure MSE + cosine similarity
 *
 * Scope: this test only exercises scalar quant/dequant fidelity on the CPU
 * reference path. It does NOT cover:
 *   - the CUDA quant kernels in ggml-cuda/turbo-quant-cuda.cuh
 *   - the set-rows pointer arithmetic in ggml-cuda/set-rows.cu (the kind of
 *     bug fixed in C3 would not be caught here)
 *   - the flash-attention HelperTurbo3/4 dequant paths
 *   - the full Q/K/V + attention pipeline with FWHT rotation.
 *
 * For end-to-end correctness, run a short prompt through the model with
 *   --cache-type-k turbo4 --cache-type-v turbo4
 * and compare perplexity / next-token logits against the fp16 baseline.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Tell ggml-common.h to emit C block type declarations. Without this,
 * struct block_turbo3_0 / block_turbo4_0 are never defined and every
 * (block_turbo*_0 *) cast below fails to parse. */
#define GGML_COMMON_DECL_C
/* Use the real declarations — matching the actual int64_t / block_turbo*_0 *
 * signatures rather than ABI-compatible-on-LP64 hand-written externs. */
#include "ggml-turbo-quant.h"

static int g_fails = 0;

#define EXPECT(cond, fmt, ...) do { \
    if (!(cond)) { printf("  *** FAIL: " fmt " ***\n", ##__VA_ARGS__); g_fails++; } \
} while (0)

static float report(const char * name, const float * input, const float * output, int d) {
    float mse = 0, cosv = 0, ni = 0, no = 0;
    for (int i = 0; i < d; i++) {
        float diff = input[i] - output[i];
        mse += diff * diff;
        cosv += input[i] * output[i];
        ni += input[i] * input[i];
        no += output[i] * output[i];
    }
    float cosine = (ni > 0 && no > 0) ? cosv / sqrtf(ni) / sqrtf(no) : 0;
    printf("  %s: MSE=%.8f  Cosine=%.6f  InNorm=%.4f  OutNorm=%.4f\n",
           name, mse / d, cosine, sqrtf(ni), sqrtf(no));
    /* Cosine similarity gate: skip if input is essentially zero */
    if (ni > 0.01f) {
        EXPECT(cosine > 0.90f, "%s cosine %.4f <= 0.90", name, cosine);
    }
    return cosine;
}

int main(void) {
    const int d = 128;
    /* Buffers — turbo4_0 block is 66 bytes, turbo3_0 is 14 bytes × 4 blocks = 56 bytes.
     * qbuf2/input_big/output_big are for the 2-block (256-elem) test. */
    char qbuf[256], qbuf2[512];
    float input[128], output[128], input2x[128];
    float input_big[256], output_big[256];
    unsigned int seed, seed2;
    float out_norm4, out_norm3, n1, n2, ratio, cos_big;
    int i;

    printf("=== TurboQuant Round-Trip Test (FWHT) ===\n\n");

    /* Populate ggml_table_f32_f16[] — GGML_FP16_TO_FP32 reads this table,
     * and it's zero-initialized until ggml_init() fills it. Without this
     * call every dequant returns 0 because the norm field maps to 0. */
    {
        struct ggml_init_params p = { 0, NULL, true };
        struct ggml_context * ctx = ggml_init(p);
        if (ctx) ggml_free(ctx);
    }

    /* Test 1: basis vector e0 */
    printf("Test 1: e0 = [1, 0, ...]\n");
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo4_0_ref(input, (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, (block_turbo3_0 *)qbuf, d);
    dequantize_row_turbo3_0((const block_turbo3_0 *)qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 2: sinusoidal with large norm */
    printf("Test 2: sin(i*0.1+0.5) * 10\n");
    for (int i = 0; i < d; i++) input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
    quantize_row_turbo4_0_ref(input, (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, (block_turbo3_0 *)qbuf, d);
    dequantize_row_turbo3_0((const block_turbo3_0 *)qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 3: cosine pattern */
    printf("Test 3: cos(i*0.2) * 5\n");
    for (int i = 0; i < d; i++) input[i] = cosf(i * 0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, (block_turbo3_0 *)qbuf, d);
    dequantize_row_turbo3_0((const block_turbo3_0 *)qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 4: random-ish data */
    printf("Test 4: pseudo-random (LCG)\n");
    seed = 12345;
    for (i = 0; i < d; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = ((float)(seed >> 16) / 32768.0f - 1.0f) * 3.0f;
    }
    quantize_row_turbo4_0_ref(input, (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, (block_turbo3_0 *)qbuf, d);
    dequantize_row_turbo3_0((const block_turbo3_0 *)qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 5: near-zero vector must NOT amplify (edge case) */
    printf("Test 5: near-zero input stays near-zero\n");
    for (i = 0; i < d; i++) input[i] = 1e-8f * (i % 3 - 1);
    quantize_row_turbo4_0_ref(input, (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    out_norm4 = 0;
    for (i = 0; i < d; i++) out_norm4 += output[i] * output[i];
    out_norm4 = sqrtf(out_norm4);
    printf("  turbo4: OutNorm=%.12f\n", out_norm4);
    EXPECT(out_norm4 < 1e-5f, "turbo4 near-zero amplified to %.3e", (double)out_norm4);
    quantize_row_turbo3_0_ref(input, (block_turbo3_0 *)qbuf, d);
    dequantize_row_turbo3_0((const block_turbo3_0 *)qbuf, output, d);
    out_norm3 = 0;
    for (i = 0; i < d; i++) out_norm3 += output[i] * output[i];
    out_norm3 = sqrtf(out_norm3);
    printf("  turbo3: OutNorm=%.12f\n", out_norm3);
    EXPECT(out_norm3 < 1e-5f, "turbo3 near-zero amplified to %.3e", (double)out_norm3);
    printf("\n");

    /* Test 6: scaling invariance — doubling the input should double output norm
     * within tolerance (confirms norm is stored and recovered correctly). */
    printf("Test 6: norm scaling — double input gives ~double output norm\n");
    seed2 = 98765;
    for (i = 0; i < d; i++) {
        seed2 = seed2 * 1103515245 + 12345;
        input[i]   = ((float)(seed2 >> 16) / 32768.0f - 1.0f) * 2.0f;
        input2x[i] = 2.0f * input[i];
    }
    quantize_row_turbo4_0_ref(input,   (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    n1 = 0;
    for (i = 0; i < d; i++) n1 += output[i] * output[i];
    n1 = sqrtf(n1);
    quantize_row_turbo4_0_ref(input2x, (block_turbo4_0 *)qbuf, d);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf, output, d);
    n2 = 0;
    for (i = 0; i < d; i++) n2 += output[i] * output[i];
    n2 = sqrtf(n2);
    ratio = n1 > 0 ? n2 / n1 : 0;
    printf("  turbo4: OutNorm(x)=%.4f  OutNorm(2x)=%.4f  ratio=%.4f (expect ~2)\n", n1, n2, ratio);
    EXPECT(ratio > 1.95f && ratio < 2.05f, "turbo4 norm ratio %.4f outside [1.95, 2.05]", ratio);
    printf("\n");

    /* Test 7: 256-element (2-block) round-trip — exercises the per-block loop. */
    printf("Test 7: 256-element (2-block) round-trip\n");
    for (i = 0; i < 256; i++) input_big[i] = sinf(i * 0.15f) * 4.0f + cosf(i * 0.07f);
    quantize_row_turbo4_0_ref(input_big, (block_turbo4_0 *)qbuf2, 256);
    dequantize_row_turbo4_0((const block_turbo4_0 *)qbuf2, output_big, 256);
    cos_big = report("turbo4 (2 blocks)", input_big, output_big, 256);
    EXPECT(cos_big > 0.92f, "turbo4 2-block cosine %.4f too low", cos_big);
    printf("\n");

    printf("=== Done: %d failures ===\n", g_fails);
    return g_fails > 0 ? 1 : 0;
}
