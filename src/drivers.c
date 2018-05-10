// Driver routines for synthesis and analysis of harmonic polynomial transforms.

#include "fasttransforms.h"

void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel
    for (int m = 2 + omp_get_thread_num(); m <= M/2; m += omp_get_num_threads())
        kernel2_sph_hi2lo(RP, m, A+(RP->n)*(2*m-1), A+(RP->n)*(2*m));
}

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

static inline void apply_givens_2x1(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A, double * B) {
    register double s1, c1;
    register double a1, a2;
    s1 = s(l, m);
    c1 = c(l, m);

    a1 = A[l];
    a2 = A[l+inc];

    A[l    ] = c1*a1 + s1*a2;
    A[l+inc] = c1*a2 - s1*a1;

    a1 = B[l];
    a2 = B[l+inc];

    B[l    ] = c1*a1 + s1*a2;
    B[l+inc] = c1*a2 - s1*a1;
}

void cache_oblivious_execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M) {
    int n = RP->n;
    for (int step = 4; step <= (M+1)/2; step *= 2) {
        for (int m = step/2; m <= M/2; m += step)
            for (int mu = m; mu < m+step/2; mu += 2)
                for (int j = m-2; j >= m-step/2; j -= 2)
                    for (int l = n-3-j; l >= 0; l--)
                        apply_givens_2x1(2, RP->s, RP->c, n, l, j, A+n*(2*mu-1), A+n*(2*mu));
        for (int m = step/2+1; m <= M/2; m += step)
            for (int mu = m; mu < m+step/2; mu += 2)
                for (int j = m-2; j >= m-step/2; j -= 2)
                    for (int l = n-3-j; l >= 0; l--)
                        apply_givens_2x1(2, RP->s, RP->c, n, l, j, A+n*(2*mu-1), A+n*(2*mu));
    }
}

void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel
    for (int m = 2 + omp_get_thread_num(); m <= M/2; m += omp_get_num_threads())
        kernel2_sph_lo2hi(RP, m, A+(RP->n)*(2*m-1), A+(RP->n)*(2*m));
}

void execute_tri_hi2lo(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel
    for (int m = 1 + omp_get_thread_num(); m < M; m += omp_get_num_threads())
        kernel1_tri_hi2lo(RP, m, A+(RP->n)*m);
}

void execute_tri_lo2hi(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel
    for (int m = 1 + omp_get_thread_num(); m < M; m += omp_get_num_threads())
        kernel1_tri_lo2hi(RP, m, A+(RP->n)*m);
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
