// Driver routines for synthesis and analysis of harmonic polynomial transforms.

#include "fasttransforms.h"

void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2 + omp_get_thread_num(); m <= M/2; m += omp_get_num_threads()) {
        kernel_sph_hi2lo(RP, m, A + N*(2*m-1));
        kernel_sph_hi2lo(RP, m, A + N*(2*m));
    }
}

void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M) {
    int N = RP->n;
    #pragma omp parallel
    for (int m = 2 + omp_get_thread_num(); m <= M/2; m += omp_get_num_threads()) {
        kernel_sph_lo2hi(RP, m, A + N*(2*m-1));
        kernel_sph_lo2hi(RP, m, A + N*(2*m));
    }
}

void execute_sph_hi2lo_SSE(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_sph_SSE(A, B, N, M);
    #pragma omp parallel
    for (int m = 2 + omp_get_thread_num(); m <= M/2; m += omp_get_num_threads())
        kernel_sph_hi2lo_SSE(RP, m, B + N*(2*m-1));
    permute_t_sph_SSE(A, B, N, M);
}

void execute_sph_lo2hi_SSE(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_sph_SSE(A, B, N, M);
    #pragma omp parallel
    for (int m = 2 + omp_get_thread_num(); m <= M/2; m += omp_get_num_threads())
        kernel_sph_lo2hi_SSE(RP, m, B + N*(2*m-1));
    permute_t_sph_SSE(A, B, N, M);
}

void execute_sph_hi2lo_AVX(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    two_warp(A, N, M);
    permute_sph_AVX(A, B, N, M);
    for (int m = 2; m <= (M%8)/2; m++)
        kernel_sph_hi2lo_SSE(RP, m, B + N*(2*m-1));
    #pragma omp parallel
    for (int m = (M%8+1)/2 + 4*omp_get_thread_num(); m <= M/2; m += 4*omp_get_num_threads()) {
        kernel_sph_hi2lo_AVX(RP, m, B + N*(2*m-1));
        kernel_sph_hi2lo_AVX(RP, m+1, B + N*(2*m+3));
    }
    permute_t_sph_AVX(A, B, N, M);
    two_warp(A, N, M);
}

void execute_sph_lo2hi_AVX(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    two_warp(A, N, M);
    permute_sph_AVX(A, B, N, M);
    for (int m = 2; m <= (M%8)/2; m++)
        kernel_sph_lo2hi_SSE(RP, m, B + N*(2*m-1));
    #pragma omp parallel
    for (int m = (M%8+1)/2 + 4*omp_get_thread_num(); m <= M/2; m += 4*omp_get_num_threads()) {
        kernel_sph_lo2hi_AVX(RP, m, B + N*(2*m-1));
        kernel_sph_lo2hi_AVX(RP, m+1, B + N*(2*m+3));
    }
    permute_t_sph_AVX(A, B, N, M);
    two_warp(A, N, M);
}

void execute_sph_hi2lo_AVX512(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    int M_star = M%16;
    four_warp(A, N, M);
    two_warp(A, N, M_star);
    permute_sph_AVX512(A, B, N, M);
    for (int m = 2; m <= (M_star%8)/2; m++)
        kernel_sph_hi2lo_SSE(RP, m, B + N*(2*m-1));
    
    #pragma omp parallel
    for (int m = (M_star%8+1)/2 + 4*omp_get_thread_num(); m <= M_star/2; m += 4*omp_get_num_threads()) {
        kernel_sph_hi2lo_AVX(RP, m, B + N*(2*m-1));
        kernel_sph_hi2lo_AVX(RP, m+1, B + N*(2*m+3));
    }
    #pragma omp parallel
    for (int m = (M_star+1)/2 + 8*omp_get_thread_num(); m <= M/2; m += 8*omp_get_num_threads()) {
        kernel_sph_hi2lo_AVX512(RP, m, B + N*(2*m-1));
        kernel_sph_hi2lo_AVX512(RP, m+1, B + N*(2*m+7));
    }
    permute_t_sph_AVX512(A, B, N, M);
    two_warp(A, N, M_star);
    reverse_four_warp(A, N, M);
}

void execute_sph_lo2hi_AVX512(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    int M_star = M%16;
    four_warp(A, N, M);
    two_warp(A, N, M_star);
    permute_sph_AVX512(A, B, N, M);
    for (int m = 2; m <= (M_star%8)/2; m++)
        kernel_sph_lo2hi_SSE(RP, m, B + N*(2*m-1));
    
    #pragma omp parallel
    for (int m = (M_star%8+1)/2 + 4*omp_get_thread_num(); m <= M_star/2; m += 4*omp_get_num_threads()) {
        kernel_sph_lo2hi_AVX(RP, m, B + N*(2*m-1));
        kernel_sph_lo2hi_AVX(RP, m+1, B + N*(2*m+3));
    }
    #pragma omp parallel
    for (int m = (M_star+1)/2 + 8*omp_get_thread_num(); m <= M/2; m += 8*omp_get_num_threads()) {
        kernel_sph_lo2hi_AVX512(RP, m, B + N*(2*m-1));
        kernel_sph_lo2hi_AVX512(RP, m+1, B + N*(2*m+7));
    }
    permute_t_sph_AVX512(A, B, N, M);
    two_warp(A, N, M_star);
    reverse_four_warp(A, N, M);
}


void execute_tri_hi2lo(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel
    for (int m = 1 + omp_get_thread_num(); m < M; m += omp_get_num_threads())
        kernel_tri_hi2lo(RP, m, A+(RP->n)*m);
}

void execute_tri_lo2hi(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel
    for (int m = 1 + omp_get_thread_num(); m < M; m += omp_get_num_threads())
        kernel_tri_lo2hi(RP, m, A+(RP->n)*m);
}

void execute_tri_hi2lo_SSE(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_tri_SSE(A, B, N, M);
    #pragma omp parallel
    for (int m = 2*omp_get_thread_num(); m < M; m += 2*omp_get_num_threads())
        kernel_tri_hi2lo_SSE(RP, m, B+N*m);
    permute_t_tri_SSE(A, B, N, M);
}

void execute_tri_lo2hi_SSE(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_tri_SSE(A, B, N, M);
    #pragma omp parallel
    for (int m = 2*omp_get_thread_num(); m < M; m += 2*omp_get_num_threads())
        kernel_tri_lo2hi_SSE(RP, m, B+N*m);
    permute_t_tri_SSE(A, B, N, M);
}

void execute_tri_hi2lo_AVX(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_tri_AVX(A, B, N, M);
    #pragma omp parallel
    for (int m = 4*omp_get_thread_num(); m < M; m += 4*omp_get_num_threads())
        kernel_tri_hi2lo_AVX(RP, m, B+N*m);
    permute_t_tri_AVX(A, B, N, M);
}

void execute_tri_lo2hi_AVX(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_tri_AVX(A, B, N, M);
    #pragma omp parallel
    for (int m = 4*omp_get_thread_num(); m < M; m += 4*omp_get_num_threads())
        kernel_tri_lo2hi_AVX(RP, m, B+N*m);
    permute_t_tri_AVX(A, B, N, M);
}

void execute_tri_hi2lo_AVX512(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_tri_AVX512(A, B, N, M);
    #pragma omp parallel
    for (int m = 8*omp_get_thread_num(); m < M; m += 8*omp_get_num_threads())
        kernel_tri_hi2lo_AVX512(RP, m, B+N*m);
    permute_t_tri_AVX512(A, B, N, M);
}

void execute_tri_lo2hi_AVX512(const RotationPlan * RP, double * A, double * B, const int M) {
    int N = RP->n;
    permute_tri_AVX512(A, B, N, M);
    #pragma omp parallel
    for (int m = 8*omp_get_thread_num(); m < M; m += 8*omp_get_num_threads())
        kernel_tri_lo2hi_AVX512(RP, m, B+N*m);
    permute_t_tri_AVX512(A, B, N, M);
}

SphericalHarmonicPlan * plan_sph2fourier(const int n) {
    SphericalHarmonicPlan * P = malloc(sizeof(SphericalHarmonicPlan));
    P->RP = plan_rotsphere(n);
    P->P1 = plan_leg2cheb(1, 0, n);
    P->P2 = plan_ultra2ultra(1, 0, n, 1.5, 1.0);
    P->P1inv = plan_cheb2leg(0, 1, n);
    P->P2inv = plan_ultra2ultra(0, 1, n, 1.0, 1.5);
    return P;
}

void execute_sph2fourier(const SphericalHarmonicPlan * P, double * A, const int N, const int M) {
    execute_sph_hi2lo(P->RP, A, M);
    cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, N, P->P1, N, A, 1);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P2, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P2, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P1, N, A+3*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M-1)/4, 1.0, P->P1, N, A+4*N, 4*N);
}

void execute_fourier2sph(const SphericalHarmonicPlan * P, double * A, const int N, const int M) {
    cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, N, P->P1inv, N, A, 1);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+2)/4, 1.0, P->P2inv, N, A+N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M+1)/4, 1.0, P->P2inv, N, A+2*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M/4, 1.0, P->P1inv, N, A+3*N, 4*N);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, (M-1)/4, 1.0, P->P1inv, N, A+4*N, 4*N);
    execute_sph_lo2hi(P->RP, A, M);
}

TriangularHarmonicPlan * plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma) {
    TriangularHarmonicPlan * P = malloc(sizeof(TriangularHarmonicPlan));
    P->RP = plan_rottriangle(n, alpha, beta, gamma);
    P->P1 = plan_jac2jac(1, 1, n, beta + gamma + 1.0, alpha, -0.5);
    P->P2 = plan_jac2jac(1, 1, n, alpha, -0.5, -0.5);
    P->P3 = plan_jac2jac(1, 1, n, gamma, beta, -0.5);
    P->P4 = plan_jac2jac(1, 1, n, beta, -0.5, -0.5);
    P->P1inv = plan_jac2jac(1, 1, n, -0.5, alpha, beta + gamma + 1.0);
    P->P2inv = plan_jac2jac(1, 1, n, -0.5, -0.5, alpha);
    P->P3inv = plan_jac2jac(1, 1, n, -0.5, beta, gamma);
    P->P4inv = plan_jac2jac(1, 1, n, -0.5, -0.5, beta);
    P->alpha = alpha;
    P->beta = beta;
    P->gamma = gamma;
    return P;
}

void execute_tri2cheb(const TriangularHarmonicPlan * P, double * A, const int N, const int M) {
    execute_tri_hi2lo(P->RP, A, M);
    if (P->beta + P->gamma != -1.5)
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M, 1.0, P->P1, N, A, N);
    if (P->alpha != -0.5) {
        alternate_sign(A, N*M);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M, 1.0, P->P2, N, A, N);
        alternate_sign(A, N*M);
    }
    if (P->gamma != -0.5)
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, M, 1.0, P->P3, N, A, N);
    if (P->beta != -0.5) {
        alternate_sign(A, N*M);
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, M, 1.0, P->P4, N, A, N);
        alternate_sign(A, N*M);
    }
    chebyshev_normalization(A, N, M);
}

void execute_cheb2tri(const TriangularHarmonicPlan * P, double * A, const int N, const int M) {
    chebyshev_normalization_t(A, N, M);
    if (P->beta != -0.5) {
        alternate_sign(A, N*M);
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, M, 1.0, P->P4inv, N, A, N);
        alternate_sign(A, N*M);
    }
    if (P->gamma != -0.5)
        cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, N, M, 1.0, P->P3inv, N, A, N);
    if (P->alpha != -0.5) {
        alternate_sign(A, N*M);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M, 1.0, P->P2inv, N, A, N);
        alternate_sign(A, N*M);
    }
    if (P->beta + P->gamma != -1.5)
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, M, 1.0, P->P1inv, N, A, N);
    execute_tri_lo2hi(P->RP, A, M);
}

static void alternate_sign(double * A, const int N) {
    for (int i = 0; i < N; i += 2)
        A[i] = -A[i];
}

static void chebyshev_normalization(double * A, const int N, const int M) {
    A[0] *= M_1_PI;
    for (int i = 1; i < N; i++)
        A[i] *= M_SQRT2*M_1_PI;
    for (int j = 1; j < M; j++)
        A[j*N] *= M_SQRT2*M_1_PI;
    for (int i = 1; i < N; i++)
        for (int j = 1; j < M; j++)
            A[i+j*N] *= M_2_PI;
}

static void chebyshev_normalization_t(double * A, const int N, const int M) {
    A[0] *= M_PI;
    for (int i = 1; i < N; i++)
        A[i] *= M_SQRT1_2*M_PI;
    for (int j = 1; j < M; j++)
        A[j*N] *= M_SQRT1_2*M_PI;
    for (int i = 1; i < N; i++)
        for (int j = 1; j < M; j++)
            A[i+j*N] *= M_PI_2;
}
