// Driver routines for synthesis and analysis of harmonic polynomial transforms.

#include "drivers.h"

void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel for schedule(dynamic)
    for (int m = 2; m <= M/2; m++)
        kernel2_sph_hi2lo(RP, A+(RP->n)*(2*m-1), A+(RP->n)*(2*m), m);
}

void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M) {
    #pragma omp parallel for schedule(dynamic)
    for (int m = 2; m <= M/2; m++)
        kernel2_sph_lo2hi(RP, A+(RP->n)*(2*m-1), A+(RP->n)*(2*m), m);
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
