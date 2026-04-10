/*
 * TurboQuant round-trip test: quantize → dequantize → measure MSE + cosine similarity
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Forward declarations — these are in ggml-turbo-quant.c */
extern void quantize_row_turbo3_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo3_0(const void * x, float * y, long long k);
extern void quantize_row_turbo4_0_ref(const float * x, void * y, long long k);
extern void dequantize_row_turbo4_0(const void * x, float * y, long long k);

static void report(const char * name, const float * input, const float * output, int d) {
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

    /* Fail if cosine similarity is too low (should be > 0.95 for good quant) */
    if (cosine < 0.90f && ni > 0.01f) {
        printf("  *** FAIL: cosine %.4f < 0.90 ***\n", cosine);
    }
}

int main(void) {
    const int d = 128;
    /* Buffers — turbo4_0 block is 66 bytes, turbo3_0 is 14 bytes × 4 blocks = 56 bytes */
    char qbuf[256];
    float input[128], output[128];

    printf("=== TurboQuant Round-Trip Test (FWHT) ===\n\n");

    /* Test 1: basis vector e0 */
    printf("Test 1: e0 = [1, 0, ...]\n");
    memset(input, 0, sizeof(input));
    input[0] = 1.0f;
    quantize_row_turbo4_0_ref(input, qbuf, d);
    dequantize_row_turbo4_0(qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, qbuf, d);
    dequantize_row_turbo3_0(qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 2: sinusoidal with large norm */
    printf("Test 2: sin(i*0.1+0.5) * 10\n");
    for (int i = 0; i < d; i++) input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
    quantize_row_turbo4_0_ref(input, qbuf, d);
    dequantize_row_turbo4_0(qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, qbuf, d);
    dequantize_row_turbo3_0(qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 3: cosine pattern */
    printf("Test 3: cos(i*0.2) * 5\n");
    for (int i = 0; i < d; i++) input[i] = cosf(i * 0.2f) * 5.0f;
    quantize_row_turbo4_0_ref(input, qbuf, d);
    dequantize_row_turbo4_0(qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, qbuf, d);
    dequantize_row_turbo3_0(qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 4: random-ish data */
    printf("Test 4: pseudo-random (LCG)\n");
    unsigned int seed = 12345;
    for (int i = 0; i < d; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = ((float)(seed >> 16) / 32768.0f - 1.0f) * 3.0f;
    }
    quantize_row_turbo4_0_ref(input, qbuf, d);
    dequantize_row_turbo4_0(qbuf, output, d);
    report("turbo4", input, output, d);
    quantize_row_turbo3_0_ref(input, qbuf, d);
    dequantize_row_turbo3_0(qbuf, output, d);
    report("turbo3", input, output, d);
    printf("\n");

    /* Test 5: near-zero vector (edge case) */
    printf("Test 5: near-zero (1e-8)\n");
    for (int i = 0; i < d; i++) input[i] = 1e-8f * (i % 3 - 1);
    quantize_row_turbo4_0_ref(input, qbuf, d);
    dequantize_row_turbo4_0(qbuf, output, d);
    printf("  turbo4: OutNorm=%.12f (should be ~0)\n", 0.0);
    float out_norm = 0;
    for (int i = 0; i < d; i++) out_norm += output[i] * output[i];
    printf("  turbo4: actual OutNorm=%.12f\n", sqrtf(out_norm));
    printf("\n");

    printf("=== Done ===\n");
    return 0;
}
