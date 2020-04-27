// Driver routines for the harmonic polynomial connection problem.

#include "fasttransforms.h"
#include "ftinternal.h"

static void chebyshev_normalization_2d(double * A, const int N, const int M) {
    for (int i = 0; i < N; i++)
        A[i] *= M_SQRT1_2;
    for (int j = 0; j < M; j++) {
        A[j*N] *= M_SQRT1_2;
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_2_PI;
    }
}

static void chebyshev_normalization_2d_t(double * A, const int N, const int M) {
    for (int i = 0; i < N; i++)
        A[i] *= M_SQRT2;
    for (int j = 0; j < M; j++) {
        A[j*N] *= M_SQRT2;
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_PI_2;
    }
}

static void chebyshev_normalization_3d(double * A, const int N, const int L, const int M) {
    for (int j = 0; j < L; j++)
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_SQRT1_2;
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N; i++)
            A[i+k*L*N] *= M_SQRT1_2;
        for (int j = 0; j < L; j++) {
            A[(j+k*L)*N] *= M_SQRT1_2;
            for (int i = 0; i < N; i++)
                A[i+(j+k*L)*N] *= M_2_PI_POW_1P5;
        }
    }
}

static void chebyshev_normalization_3d_t(double * A, const int N, const int L, const int M) {
    for (int j = 0; j < L; j++)
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_SQRT2;
    for (int k = 0; k < M; k++) {
        for (int i = 0; i < N; i++)
            A[i+k*L*N] *= M_SQRT2;
        for (int j = 0; j < L; j++) {
            A[(j+k*L)*N] *= M_SQRT2;
            for (int i = 0; i < N; i++)
                A[i+(j+k*L)*N] *= M_PI_2_POW_1P5;
        }
    }
}

static void partial_chebyshev_normalization(double * A, const int N, const int M) {
    for (int j = 1; j < M; j += 4)
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_2_PI_POW_0P5;
    for (int j = 2; j < M; j += 4)
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_2_PI_POW_0P5;
}

static void partial_chebyshev_normalization_t(double * A, const int N, const int M) {
    for (int j = 1; j < M; j += 4)
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_PI_2_POW_0P5;
    for (int j = 2; j < M; j += 4)
        for (int i = 0; i < N; i++)
            A[i+j*N] *= M_PI_2_POW_0P5;
}

void ft_set_num_threads(const int n) {FT_SET_NUM_THREADS(n);}

void ft_execute_sph_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_sph_hi2lo_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_sph_hi2lo_AVX_FMA(RP, A, B, M);
        else
            return execute_sph_hi2lo_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_sph_hi2lo_SSE2(RP, A, B, M);
    else
        return execute_sph_hi2lo_default(RP, A, M);
}

void ft_execute_sph_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_sph_lo2hi_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_sph_lo2hi_AVX_FMA(RP, A, B, M);
        else
            return execute_sph_lo2hi_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_sph_lo2hi_SSE2(RP, A, B, M);
    else
        return execute_sph_lo2hi_default(RP, A, M);
}

void execute_sph_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_sph_hi2lo_default(RP, m%2, m, A + N*(2*m-1), 1);
        kernel_sph_hi2lo_default(RP, m%2, m, A + N*(2*m), 1);
    }
}

void execute_sph_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_sph_lo2hi_default(RP, m%2, m, A + N*(2*m-1), 1);
        kernel_sph_lo2hi_default(RP, m%2, m, A + N*(2*m), 1);
    }
}

#define EXECUTE_SPH(S, KERNEL_DEFAULT, KERNEL_SIMD)                            \
int N = RP->n;                                                                 \
warp(A, N, M, S/2);                                                            \
permute_sph(A, B, N, M, S);                                                    \
for (int m = 2; m <= (M%(2*S))/2; m++) {                                       \
    KERNEL_DEFAULT(RP, m%2, m, B+N*(2*m-1), 1);                                \
    KERNEL_DEFAULT(RP, m%2, m, B+N*(2*m), 1);                                  \
}                                                                              \
_Pragma("omp parallel")                                                        \
for (int m = (M%(2*S)+1)/2 + S*FT_GET_THREAD_NUM(); m <= M/2; m += S*FT_GET_NUM_THREADS()) { \
    KERNEL_SIMD(RP, m%2, m, B+N*(2*m-1), S);                                   \
    KERNEL_SIMD(RP, (m+1)%2, m+1, B+N*(2*m-1+S), S);                           \
}                                                                              \
permute_t_sph(A, B, N, M, S);                                                  \
warp_t(A, N, M, S/2);

void execute_sph_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(2, kernel_sph_hi2lo_default, kernel_sph_hi2lo_SSE2)
}

void execute_sph_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(2, kernel_sph_lo2hi_default, kernel_sph_lo2hi_SSE2)
}

void execute_sph_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(4, kernel_sph_hi2lo_default, kernel_sph_hi2lo_AVX)
}

void execute_sph_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(4, kernel_sph_lo2hi_default, kernel_sph_lo2hi_AVX)
}

void execute_sph_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(4, kernel_sph_hi2lo_default, kernel_sph_hi2lo_AVX_FMA)
}

void execute_sph_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(4, kernel_sph_lo2hi_default, kernel_sph_lo2hi_AVX_FMA)
}

void execute_sph_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(8, kernel_sph_hi2lo_default, kernel_sph_hi2lo_AVX512F)
}

void execute_sph_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPH(8, kernel_sph_lo2hi_default, kernel_sph_lo2hi_AVX512F)
}


void ft_execute_sphv_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_sphv_hi2lo_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_sphv_hi2lo_AVX_FMA(RP, A, B, M);
        else
            return execute_sphv_hi2lo_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_sphv_hi2lo_SSE2(RP, A, B, M);
    else
        return execute_sphv_hi2lo_default(RP, A, M);
}

void ft_execute_sphv_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_sphv_lo2hi_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_sphv_lo2hi_AVX_FMA(RP, A, B, M);
        else
            return execute_sphv_lo2hi_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_sphv_lo2hi_SSE2(RP, A, B, M);
    else
        return execute_sphv_lo2hi_default(RP, A, M);
}

void execute_sphv_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2+FT_GET_THREAD_NUM(); m <= M/2-1; m += FT_GET_NUM_THREADS()) {
        kernel_sph_hi2lo_default(RP, m%2, m, A + N*(2*m+1), 1);
        kernel_sph_hi2lo_default(RP, m%2, m, A + N*(2*m+2), 1);
    }
}

void execute_sphv_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2+FT_GET_THREAD_NUM(); m <= M/2-1; m += FT_GET_NUM_THREADS()) {
        kernel_sph_lo2hi_default(RP, m%2, m, A + N*(2*m+1), 1);
        kernel_sph_lo2hi_default(RP, m%2, m, A + N*(2*m+2), 1);
    }
}

#define EXECUTE_SPHV(S, KERNEL_DEFAULT, KERNEL_SIMD)                           \
int N = RP->n;                                                                 \
warp(A+2*N, N, M-2, S/2);                                                      \
permute_sph(A+2*N, B+2*N, N, M-2, S);                                          \
for (int m = 2; m <= ((M-2)%(2*S))/2; m++) {                                   \
    KERNEL_DEFAULT(RP, m%2, m, B+N*(2*m+1), 1);                                \
    KERNEL_DEFAULT(RP, m%2, m, B+N*(2*m+2), 1);                                \
}                                                                              \
_Pragma("omp parallel")                                                        \
for (int m = ((M-2)%(2*S)+1)/2 + S*FT_GET_THREAD_NUM(); m <= M/2-1; m += S*FT_GET_NUM_THREADS()) { \
    KERNEL_SIMD(RP, m%2, m, B+N*(2*m+1), S);                                   \
    KERNEL_SIMD(RP, (m+1)%2, m+1, B+N*(2*m+1+S), S);                           \
}                                                                              \
permute_t_sph(A+2*N, B+2*N, N, M-2, S);                                        \
warp_t(A+2*N, N, M-2, S/2);

void execute_sphv_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(2, kernel_sph_hi2lo_default, kernel_sph_hi2lo_SSE2)
}

void execute_sphv_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(2, kernel_sph_lo2hi_default, kernel_sph_lo2hi_SSE2)
}

void execute_sphv_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(4, kernel_sph_hi2lo_default, kernel_sph_hi2lo_AVX)
}

void execute_sphv_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(4, kernel_sph_lo2hi_default, kernel_sph_lo2hi_AVX)
}

void execute_sphv_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(4, kernel_sph_hi2lo_default, kernel_sph_hi2lo_AVX_FMA)
}

void execute_sphv_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(4, kernel_sph_lo2hi_default, kernel_sph_lo2hi_AVX_FMA)
}

void execute_sphv_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(8, kernel_sph_hi2lo_default, kernel_sph_hi2lo_AVX512F)
}

void execute_sphv_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_SPHV(8, kernel_sph_lo2hi_default, kernel_sph_lo2hi_AVX512F)
}


void ft_execute_tri_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_tri_hi2lo_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_tri_hi2lo_AVX_FMA(RP, A, B, M);
        else
            return execute_tri_hi2lo_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_tri_hi2lo_SSE2(RP, A, B, M);
    else
        return execute_tri_hi2lo_default(RP, A, M);
}

void ft_execute_tri_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_tri_lo2hi_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_tri_lo2hi_AVX_FMA(RP, A, B, M);
        else
            return execute_tri_lo2hi_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_tri_lo2hi_SSE2(RP, A, B, M);
    else
        return execute_tri_lo2hi_default(RP, A, M);
}

void execute_tri_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 1+FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS())
        kernel_tri_hi2lo_default(RP, 0, m, A+N*m, 1);
}

void execute_tri_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 1+FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS())
        kernel_tri_lo2hi_default(RP, 0, m, A+N*m, 1);
}

#define EXECUTE_TRI(S, KERNEL_DEFAULT, KERNEL_SIMD)                            \
int N = RP->n;                                                                 \
permute_tri(A, B, N, M, S);                                                    \
for (int m = 1; m < M%S; m++)                                                  \
    KERNEL_DEFAULT(RP, 0, m, B+N*m, 1);                                        \
_Pragma("omp parallel")                                                        \
for (int m = M%S + S*FT_GET_THREAD_NUM(); m < M; m += S*FT_GET_NUM_THREADS())  \
    KERNEL_SIMD(RP, 0, m, B+N*m, S);                                           \
permute_t_tri(A, B, N, M, S);

void execute_tri_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(2, kernel_tri_hi2lo_default, kernel_tri_hi2lo_SSE2)
}

void execute_tri_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(2, kernel_tri_lo2hi_default, kernel_tri_lo2hi_SSE2)
}

void execute_tri_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(4, kernel_tri_hi2lo_default, kernel_tri_hi2lo_AVX)
}

void execute_tri_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(4, kernel_tri_lo2hi_default, kernel_tri_lo2hi_AVX)
}

void execute_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(4, kernel_tri_hi2lo_default, kernel_tri_hi2lo_AVX_FMA)
}

void execute_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(4, kernel_tri_lo2hi_default, kernel_tri_lo2hi_AVX_FMA)
}

void execute_tri_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(8, kernel_tri_hi2lo_default, kernel_tri_hi2lo_AVX512F)
}

void execute_tri_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_TRI(8, kernel_tri_lo2hi_default, kernel_tri_lo2hi_AVX512F)
}


void ft_execute_disk_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_disk_hi2lo_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_disk_hi2lo_AVX_FMA(RP, A, B, M);
        else
            return execute_disk_hi2lo_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_disk_hi2lo_SSE2(RP, A, B, M);
    else
        return execute_disk_hi2lo_default(RP, A, M);
}

void ft_execute_disk_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return execute_disk_lo2hi_AVX512F(RP, A, B, M);
    else if (simd.avx) {
        if (simd.fma)
            return execute_disk_lo2hi_AVX_FMA(RP, A, B, M);
        else
            return execute_disk_lo2hi_AVX(RP, A, B, M);
    }
    else if (simd.sse2)
        return execute_disk_lo2hi_SSE2(RP, A, B, M);
    else
        return execute_disk_lo2hi_default(RP, A, M);
}

void execute_disk_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_disk_hi2lo_default(RP, m%2, m, A + N*(2*m-1), 1);
        kernel_disk_hi2lo_default(RP, m%2, m, A + N*(2*m), 1);
    }
}

void execute_disk_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_disk_lo2hi_default(RP, m%2, m, A + N*(2*m-1), 1);
        kernel_disk_lo2hi_default(RP, m%2, m, A + N*(2*m), 1);
    }
}

#define EXECUTE_DISK(S, KERNEL_DEFAULT, KERNEL_SIMD)                           \
int N = RP->n;                                                                 \
warp(A, N, M, S/2);                                                            \
permute_disk(A, B, N, M, S);                                                   \
for (int m = 2; m <= (M%(2*S))/2; m++) {                                       \
    KERNEL_DEFAULT(RP, m%2, m, B+N*(2*m-1), 1);                                \
    KERNEL_DEFAULT(RP, m%2, m, B+N*(2*m), 1);                                  \
}                                                                              \
_Pragma("omp parallel")                                                        \
for (int m = (M%(2*S)+1)/2 + S*FT_GET_THREAD_NUM(); m <= M/2; m += S*FT_GET_NUM_THREADS()) { \
    KERNEL_SIMD(RP, m%2, m, B+N*(2*m-1), S);                                   \
    KERNEL_SIMD(RP, (m+1)%2, m+1, B+N*(2*m-1+S), S);                           \
}                                                                              \
permute_t_disk(A, B, N, M, S);                                                 \
warp_t(A, N, M, S/2);

void execute_disk_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(2, kernel_disk_hi2lo_default, kernel_disk_hi2lo_SSE2)
}

void execute_disk_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(2, kernel_disk_lo2hi_default, kernel_disk_lo2hi_SSE2)
}

void execute_disk_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(4, kernel_disk_hi2lo_default, kernel_disk_hi2lo_AVX)
}

void execute_disk_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(4, kernel_disk_lo2hi_default, kernel_disk_lo2hi_AVX)
}

void execute_disk_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(4, kernel_disk_hi2lo_default, kernel_disk_hi2lo_AVX_FMA)
}

void execute_disk_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(4, kernel_disk_lo2hi_default, kernel_disk_lo2hi_AVX_FMA)
}

void execute_disk_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(8, kernel_disk_hi2lo_default, kernel_disk_hi2lo_AVX512F)
}

void execute_disk_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M) {
    EXECUTE_DISK(8, kernel_disk_lo2hi_default, kernel_disk_lo2hi_AVX512F)
}


void ft_execute_tet_hi2lo(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M) {
    int N = RP1->n;
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        for (int l = 0; l < L-m; l++)
            kernel_tri_hi2lo_default(RP1, 0, l+m, A+N*(l+L*m), 1);
        ft_kernel_tet_hi2lo(RP2, L, m, A+N*L*m);
    }
}

void ft_execute_tet_lo2hi(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M) {
    int N = RP1->n;
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        ft_kernel_tet_lo2hi(RP2, L, m, A+N*L*m);
        for (int l = 0; l < L-m; l++)
            kernel_tri_lo2hi_default(RP1, 0, l+m, A+N*(l+L*m), 1);
    }
}

void execute_tet_hi2lo_SSE2(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M) {
    int N = RP1->n;
    int NB = VALIGN(N);
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        old_permute_tri(A+N*L*m, B+NB*L*m, N, L-m, 2);
        if ((L-m)%2)
            kernel_tri_hi2lo_default(RP1, 0, m, B+NB*L*m, 1);
        for (int l = (L-m)%2; l < L-m; l += 2)
            kernel_tri_hi2lo_SSE2(RP1, 0, l+m, B+NB*(l+L*m), 2);
        old_permute_t_tri(A+N*L*m, B+NB*L*m, N, L-m, 2);
        permute(A+N*L*m, B+NB*L*m, N, L, 1);
        kernel_tet_hi2lo_SSE2(RP2, L, m, B+NB*L*m);
        permute_t(A+N*L*m, B+NB*L*m, N, L, 1);
    }
}

void execute_tet_lo2hi_SSE2(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M) {
    int N = RP1->n;
    int NB = VALIGN(N);
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        permute(A+N*L*m, B+NB*L*m, N, L, 1);
        kernel_tet_lo2hi_SSE2(RP2, L, m, B+NB*L*m);
        permute_t(A+N*L*m, B+NB*L*m, N, L, 1);
        old_permute_tri(A+N*L*m, B+NB*L*m, N, L-m, 2);
        if ((L-m)%2)
            kernel_tri_lo2hi_default(RP1, 0, m, B+NB*L*m, 1);
        for (int l = (L-m)%2; l < L-m; l += 2)
            kernel_tri_lo2hi_SSE2(RP1, 0, l+m, B+NB*(l+L*m), 2);
        old_permute_t_tri(A+N*L*m, B+NB*L*m, N, L-m, 2);
    }
}

void execute_tet_hi2lo_AVX(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M) {
    int N = RP1->n;
    int NB = VALIGN(N);
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        old_permute_tri(A+N*L*m, B+NB*L*m, N, L-m, 4);
        if ((L-m)%2)
            kernel_tri_hi2lo_default(RP1, 0, m, B+NB*L*m, 1);
        for (int l = (L-m)%2; l < (L-m)%8; l += 2)
            kernel_tri_hi2lo_SSE2(RP1, 0, l+m, B+NB*(l+L*m), 2);
        for (int l = (L-m)%8; l < L-m; l += 4)
            kernel_tri_hi2lo_AVX(RP1, 0, l+m, B+NB*(l+L*m), 4);
        old_permute_t_tri(A+N*L*m, B+NB*L*m, N, L-m, 4);
        permute(A+N*L*m, B+NB*L*m, N, L, 1);
        kernel_tet_hi2lo_AVX(RP2, L, m, B+NB*L*m);
        permute_t(A+N*L*m, B+NB*L*m, N, L, 1);
    }
}

void execute_tet_lo2hi_AVX(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M) {
    int N = RP1->n;
    int NB = VALIGN(N);
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        permute(A+N*L*m, B+NB*L*m, N, L, 1);
        kernel_tet_lo2hi_AVX(RP2, L, m, B+NB*L*m);
        permute_t(A+N*L*m, B+NB*L*m, N, L, 1);
        old_permute_tri(A+N*L*m, B+NB*L*m, N, L-m, 4);
        if ((L-m)%2)
            kernel_tri_lo2hi_default(RP1, 0, m, B+NB*L*m, 1);
        for (int l = (L-m)%2; l < (L-m)%8; l += 2)
            kernel_tri_lo2hi_SSE2(RP1, 0, l+m, B+NB*(l+L*m), 2);
        for (int l = (L-m)%8; l < L-m; l += 4)
            kernel_tri_lo2hi_AVX(RP1, 0, l+m, B+NB*(l+L*m), 4);
        old_permute_t_tri(A+N*L*m, B+NB*L*m, N, L-m, 4);
    }
}

void execute_tet_hi2lo_AVX512F(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M) {
    int N = RP1->n;
    int NB = VALIGN(N);
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        old_permute_tri(A+N*L*m, B+NB*L*m, N, L-m, 8);
        if ((L-m)%2)
            kernel_tri_hi2lo_default(RP1, 0, m, B+NB*L*m, 1);
        for (int l = (L-m)%2; l < (L-m)%8; l += 2)
            kernel_tri_hi2lo_SSE2(RP1, 0, l+m, B+NB*(l+L*m), 2);
        for (int l = (L-m)%8; l < (L-m)%16; l += 4)
            kernel_tri_hi2lo_AVX(RP1, 0, l+m, B+NB*(l+L*m), 4);
        for (int l = (L-m)%16; l < L-m; l += 8)
            kernel_tri_hi2lo_AVX512F(RP1, 0, l+m, B+NB*(l+L*m), 8);
        old_permute_t_tri(A+N*L*m, B+NB*L*m, N, L-m, 8);
        permute(A+N*L*m, B+NB*L*m, N, L, 1);
        kernel_tet_hi2lo_AVX512F(RP2, L, m, B+NB*L*m);
        permute_t(A+N*L*m, B+NB*L*m, N, L, 1);
    }
}

void execute_tet_lo2hi_AVX512F(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M) {
    int N = RP1->n;
    int NB = VALIGN(N);
    #pragma omp parallel
    for (int m = FT_GET_THREAD_NUM(); m < M; m += FT_GET_NUM_THREADS()) {
        permute(A+N*L*m, B+NB*L*m, N, L, 1);
        kernel_tet_lo2hi_AVX512F(RP2, L, m, B+NB*L*m);
        permute_t(A+N*L*m, B+NB*L*m, N, L, 1);
        old_permute_tri(A+N*L*m, B+NB*L*m, N, L-m, 8);
        if ((L-m)%2)
            kernel_tri_lo2hi_default(RP1, 0, m, B+NB*L*m, 1);
        for (int l = (L-m)%2; l < (L-m)%8; l += 2)
            kernel_tri_lo2hi_SSE2(RP1, 0, l+m, B+NB*(l+L*m), 2);
        for (int l = (L-m)%8; l < (L-m)%16; l += 4)
            kernel_tri_lo2hi_AVX(RP1, 0, l+m, B+NB*(l+L*m), 4);
        for (int l = (L-m)%16; l < L-m; l += 8)
            kernel_tri_lo2hi_AVX512F(RP1, 0, l+m, B+NB*(l+L*m), 8);
        old_permute_t_tri(A+N*L*m, B+NB*L*m, N, L-m, 8);
    }
}


void ft_execute_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.sse2)
        return execute_spinsph_hi2lo_SSE2(SRP, A, M);
    else
        return execute_spinsph_hi2lo_default(SRP, A, M);
}

void ft_execute_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M) {
    ft_simd simd = get_simd();
    if (simd.sse2)
        return execute_spinsph_lo2hi_SSE2(SRP, A, M);
    else
        return execute_spinsph_lo2hi_default(SRP, A, M);
}

void execute_spinsph_hi2lo_default(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M) {
    int N = SRP->n;
    kernel_spinsph_hi2lo_default(SRP, 0, A, 1);
    #pragma omp parallel
    for (int m = 1+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_spinsph_hi2lo_default(SRP, -m, A + N*(2*m-1), 1);
        kernel_spinsph_hi2lo_default(SRP,  m, A + N*(2*m), 1);
    }
}

void execute_spinsph_lo2hi_default(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M) {
    int N = SRP->n;
    kernel_spinsph_lo2hi_default(SRP, 0, A, 1);
    #pragma omp parallel
    for (int m = 1+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_spinsph_lo2hi_default(SRP, -m, A + N*(2*m-1), 1);
        kernel_spinsph_lo2hi_default(SRP,  m, A + N*(2*m), 1);
    }
}

void execute_spinsph_hi2lo_SSE2(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M) {
    int N = SRP->n;
    kernel_spinsph_hi2lo_SSE2(SRP, 0, A, 1);
    #pragma omp parallel
    for (int m = 1+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_spinsph_hi2lo_SSE2(SRP, -m, A + N*(2*m-1), 1);
        kernel_spinsph_hi2lo_SSE2(SRP,  m, A + N*(2*m), 1);
    }
}

void execute_spinsph_lo2hi_SSE2(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M) {
    int N = SRP->n;
    kernel_spinsph_lo2hi_SSE2(SRP, 0, A, 1);
    #pragma omp parallel
    for (int m = 1+FT_GET_THREAD_NUM(); m <= M/2; m += FT_GET_NUM_THREADS()) {
        kernel_spinsph_lo2hi_SSE2(SRP, -m, A + N*(2*m-1), 1);
        kernel_spinsph_lo2hi_SSE2(SRP,  m, A + N*(2*m), 1);
    }
}


void ft_destroy_harmonic_plan(ft_harmonic_plan * P) {
    ft_destroy_rotation_plan(P->RP);
    VFREE(P->B);
    free(P->P1);
    free(P->P2);
    free(P->P1inv);
    free(P->P2inv);
    free(P);
}

ft_harmonic_plan * ft_plan_sph2fourier(const int n) {
    ft_harmonic_plan * P = malloc(sizeof(ft_harmonic_plan));
    P->RP = ft_plan_rotsphere(n);
    P->B = VMALLOC(VALIGN(n) * (2*n-1) * sizeof(double));
    P->P1 = plan_legendre_to_chebyshev(1, 0, n);
    P->P2 = plan_ultraspherical_to_ultraspherical(1, 0, n, 1.5, 1.0);
    P->P1inv = plan_chebyshev_to_legendre(0, 1, n);
    P->P2inv = plan_ultraspherical_to_ultraspherical(0, 1, n, 1.0, 1.5);
    return P;
}

void ft_execute_sph2fourier(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    ft_execute_sph_hi2lo(P->RP, A, P->B, M);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, 1.0, P->P1, N, A, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P2, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P2, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P1, N, A+3*N, 4*N);
}

void ft_execute_fourier2sph(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, 1.0, P->P1inv, N, A, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P2inv, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P2inv, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P1inv, N, A+3*N, 4*N);
    ft_execute_sph_lo2hi(P->RP, A, P->B, M);
}

void ft_execute_sphv2fourier(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    ft_execute_sphv_hi2lo(P->RP, A, P->B, M);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, 1.0, P->P2, N, A, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P1, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P1, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P2, N, A+3*N, 4*N);
}

void ft_execute_fourier2sphv(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, 1.0, P->P2inv, N, A, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P1inv, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P1inv, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P2inv, N, A+3*N, 4*N);
    ft_execute_sphv_lo2hi(P->RP, A, P->B, M);
}

ft_harmonic_plan * ft_plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma) {
    ft_harmonic_plan * P = malloc(sizeof(ft_harmonic_plan));
    P->RP = ft_plan_rottriangle(n, alpha, beta, gamma);
    P->B = VMALLOC(VALIGN(n) * n * sizeof(double));
    P->P1 = plan_jacobi_to_jacobi(1, 1, n, beta + gamma + 1.0, alpha, -0.5, -0.5);
    P->P2 = plan_jacobi_to_jacobi(1, 1, n, gamma, beta, -0.5, -0.5);
    P->P1inv = plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, beta + gamma + 1.0, alpha);
    P->P2inv = plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, gamma, beta);
    P->alpha = alpha;
    P->beta = beta;
    P->gamma = gamma;
    return P;
}

void ft_execute_tri2cheb(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    ft_execute_tri_hi2lo(P->RP, A, P->B, M);
    if ((P->beta + P->gamma != -1.5) || (P->alpha != -0.5))
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M, 1.0, P->P1, N, A, N);
    if ((P->gamma != -0.5) || (P->beta != -0.5))
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, M, 1.0, P->P2, N, A, N);
    chebyshev_normalization_2d(A, N, M);
}

void ft_execute_cheb2tri(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    chebyshev_normalization_2d_t(A, N, M);
    if ((P->beta != -0.5) || (P->gamma != -0.5))
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, M, 1.0, P->P2inv, N, A, N);
    if ((P->alpha != -0.5) || (P->beta + P->gamma != -1.5))
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M, 1.0, P->P1inv, N, A, N);
    ft_execute_tri_lo2hi(P->RP, A, P->B, M);
}

ft_harmonic_plan * ft_plan_disk2cxf(const int n) {
    ft_harmonic_plan * P = malloc(sizeof(ft_harmonic_plan));
    P->RP = ft_plan_rotdisk(n);
    P->B = VMALLOC(VALIGN(n) * (4*n-3) * sizeof(double));
    P->P1 = plan_legendre_to_chebyshev(1, 0, n);
    P->P2 = plan_jacobi_to_jacobi(1, 1, n, 0.0, 1.0, -0.5, 0.5);
    P->P1inv = plan_chebyshev_to_legendre(0, 1, n);
    P->P2inv = plan_jacobi_to_jacobi(1, 1, n, -0.5, 0.5, 0.0, 1.0);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++) {
            P->P1[i+j*n] *= 2.0;
            P->P2[i+j*n] *= 2.0;
            P->P1inv[i+j*n] *= 0.5;
            P->P2inv[i+j*n] *= 0.5;
        }
    return P;
}

void ft_execute_disk2cxf(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    ft_execute_disk_hi2lo(P->RP, A, P->B, M);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, 1.0, P->P1, N, A, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P2, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P2, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P1, N, A+3*N, 4*N);
    partial_chebyshev_normalization(A, N, M);
}

void ft_execute_cxf2disk(const ft_harmonic_plan * P, double * A, const int N, const int M) {
    partial_chebyshev_normalization_t(A, N, M);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, 1.0, P->P1inv, N, A, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P2inv, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P2inv, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P1inv, N, A+3*N, 4*N);
    ft_execute_disk_lo2hi(P->RP, A, P->B, M);
}

void ft_destroy_tetrahedral_harmonic_plan(ft_tetrahedral_harmonic_plan * P) {
    ft_destroy_rotation_plan(P->RP1);
    ft_destroy_rotation_plan(P->RP2);
    VFREE(P->B);
    free(P->P1);
    free(P->P2);
    free(P->P3);
    free(P->P1inv);
    free(P->P2inv);
    free(P->P3inv);
    free(P);
}

ft_tetrahedral_harmonic_plan * ft_plan_tet2cheb(const int n, const double alpha, const double beta, const double gamma, const double delta) {
    ft_tetrahedral_harmonic_plan * P = malloc(sizeof(ft_tetrahedral_harmonic_plan));
    P->RP1 = ft_plan_rottriangle(n, alpha, beta, gamma + delta + 1.0);
    P->RP2 = ft_plan_rottriangle(n, beta, gamma, delta);
    P->B = VMALLOC(VALIGN(n) * n * n * sizeof(double));
    P->P1 = plan_jacobi_to_jacobi(1, 1, n, beta + gamma + delta + 2.0, alpha, -0.5, -0.5);
    P->P2 = plan_jacobi_to_jacobi(1, 1, n, gamma + delta + 1.0, beta, -0.5, -0.5);
    P->P3 = plan_jacobi_to_jacobi(1, 1, n, delta, gamma, -0.5, -0.5);
    P->P1inv = plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, beta + gamma + delta + 2.0, alpha);
    P->P2inv = plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, gamma + delta + 1.0, beta);
    P->P3inv = plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, delta, gamma);
    P->alpha = alpha;
    P->beta = beta;
    P->gamma = gamma;
    P->delta = delta;
    return P;
}

void ft_execute_tet2cheb(const ft_tetrahedral_harmonic_plan * P, double * A, const int N, const int L, const int M) {
    execute_tet_hi2lo_AVX(P->RP1, P->RP2, A, P->B, L, M);
    if ((P->beta + P->gamma + P->delta != -2.5) || (P->alpha != -0.5))
        for (int m = 0; m < M; m++)
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, L, 1.0, P->P1, N, A+N*L*m, N);
    if ((P->gamma + P->delta != -1.5) || (P->beta != -0.5))
        for (int m = 0; m < M; m++)
            cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, L, 1.0, P->P2, N, A+N*L*m, N);
    if ((P->delta != -0.5) || (P->gamma != -0.5))
        for (int n = 0; n < N; n++)
            for (int l = 0; l < L; l++)
                cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, N, P->P3, N, A+n+N*l, N*L);
    chebyshev_normalization_3d(A, N, L, M);
}

void ft_execute_cheb2tet(const ft_tetrahedral_harmonic_plan * P, double * A, const int N, const int L, const int M) {
    chebyshev_normalization_3d_t(A, N, L, M);
    if ((P->gamma != -0.5) || (P->delta != -0.5))
        for (int n = 0; n < N; n++)
            for (int l = 0; l < L; l++)
                cblas_dtrmv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, N, P->P3inv, N, A+n+N*l, N*L);
    if ((P->beta != -0.5) || (P->gamma + P->delta != -1.5))
        for (int m = 0; m < M; m++)
            cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, L, 1.0, P->P2inv, N, A+N*L*m, N);
    if ((P->alpha != -0.5) || (P->beta + P->gamma + P->delta != -2.5))
        for (int m = 0; m < M; m++)
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, L, 1.0, P->P1inv, N, A+N*L*m, N);
    execute_tet_lo2hi_AVX(P->RP1, P->RP2, A, P->B, L, M);
}

void ft_destroy_spin_harmonic_plan(ft_spin_harmonic_plan * P) {
    ft_destroy_spin_rotation_plan(P->SRP);
    VFREE(P->B);
    free(P->P1);
    free(P->P2);
    free(P->P1inv);
    free(P->P2inv);
    free(P);
}

ft_spin_harmonic_plan * ft_plan_spinsph2fourier(const int n, const int s) {
    ft_spin_harmonic_plan * P = malloc(sizeof(ft_spin_harmonic_plan));
    P->SRP = ft_plan_rotspinsphere(n, s);
    P->B = VMALLOC(VALIGN(n) * (2*n-1) * sizeof(ft_complex));
    double * P1 = plan_legendre_to_chebyshev(1, 0, n);
    double * P2 = plan_ultraspherical_to_ultraspherical(1, 0, n, 1.5, 1.0);
    double * P1inv = plan_chebyshev_to_legendre(0, 1, n);
    double * P2inv = plan_ultraspherical_to_ultraspherical(0, 1, n, 1.0, 1.5);
    P->P1 = calloc(n*n, sizeof(ft_complex));
    P->P2 = calloc(n*n, sizeof(ft_complex));
    P->P1inv = calloc(n*n, sizeof(ft_complex));
    P->P2inv = calloc(n*n, sizeof(ft_complex));
    for (int i = 0; i < n*n; i++) {
        P->P1[i][0] = P1[i];
        P->P2[i][0] = P2[i];
        P->P1inv[i][0] = P1inv[i];
        P->P2inv[i][0] = P2inv[i];
    }
    free(P1);
    free(P2);
    free(P1inv);
    free(P2inv);
    P->s = s;
    return P;
}

void ft_execute_spinsph2fourier(const ft_spin_harmonic_plan * P, ft_complex * A, const int N, const int M) {
    ft_execute_spinsph_hi2lo(P->SRP, A, P->B, M);
    ft_complex alpha = {1.0, 0.0};
    if (P->s%2 == 0) {
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, &alpha, P->P1, N, A, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, &alpha, P->P2, N, A+N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, &alpha, P->P2, N, A+2*N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, &alpha, P->P1, N, A+3*N, 4*N);
    }
    else {
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, &alpha, P->P2, N, A, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, &alpha, P->P1, N, A+N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, &alpha, P->P1, N, A+2*N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, &alpha, P->P2, N, A+3*N, 4*N);
    }
}

void ft_execute_fourier2spinsph(const ft_spin_harmonic_plan * P, ft_complex * A, const int N, const int M) {
    ft_complex alpha = {1.0, 0.0};
    if (P->s%2 == 0) {
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, &alpha, P->P1inv, N, A, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, &alpha, P->P2inv, N, A+N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, &alpha, P->P2inv, N, A+2*N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, &alpha, P->P1inv, N, A+3*N, 4*N);
    }
    else {
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+3)/4, &alpha, P->P2inv, N, A, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, &alpha, P->P1inv, N, A+N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, &alpha, P->P1inv, N, A+2*N, 4*N);
        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, &alpha, P->P2inv, N, A+3*N, 4*N);
    }
    ft_execute_spinsph_lo2hi(P->SRP, A, P->B, M);
}
