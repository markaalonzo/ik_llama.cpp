#pragma once

// Canonical TurboQuant constants. Each TU (CPU, CUDA quantize, CUDA FWHT,
// iqk FA templates) declares its own storage and initializes from these
// macros so values stay in sync without duplicating the literals.

#define TURBO_WHT_S1_SIZE 128
#define TURBO_WHT_S1_INIT { -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1, \
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1, \
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1, \
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1 }

#define TURBO_WHT_S2_SIZE 128
#define TURBO_WHT_S2_INIT { 1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1, \
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, \
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, \
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1 }

#define TURBO_CENTROIDS_3BIT_SIZE 8
#define TURBO_CENTROIDS_3BIT_INIT { -0.190685f, -0.117832f, -0.065717f, -0.021460f, \
     0.021460f,  0.065717f,  0.117832f,  0.190685f }

#define TURBO_MIDPOINTS_3BIT_SIZE 7
#define TURBO_MIDPOINTS_3BIT_INIT { -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f }

#define TURBO_CENTROIDS_4BIT_SIZE 16
#define TURBO_CENTROIDS_4BIT_INIT { -0.241556f, -0.182907f, -0.143047f, -0.111065f, \
    -0.083317f, -0.058069f, -0.034311f, -0.011353f, \
     0.011353f,  0.034311f,  0.058069f,  0.083317f, \
     0.111065f,  0.143047f,  0.182907f,  0.241556f, }

#define TURBO_MIDPOINTS_4BIT_SIZE 15
#define TURBO_MIDPOINTS_4BIT_INIT { -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f, \
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f, \
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f, }

