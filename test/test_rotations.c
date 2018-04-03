#include "fasttransforms.h"
#include "utilities.h"

const int N = 1025;

int main(void) {
    double * A, * B;
    RotationPlan * RP;
    double nrm;

    printf("\n\nTesting the computation of the spherical harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        RP = plan_rotsphere(n);

        printf("\t\tThe 2-norm relative error in sqrt(s^2+c^2): %17.16e.\n", rotnorm(RP)/sqrt(n*(n+1)/2));

        A = (double *) calloc(n, sizeof(double));
        B = (double *) calloc(n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-m; i < n; i++)
                A[i] = B[i] = 0.0;
            kernel1_sph_hi2lo(RP, m, A);
            kernel1_sph_lo2hi(RP, m, A);
            nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %17.16e.\n", sqrt(nrm));

        free(A);
        free(B);

        A = (double *) calloc(2*n, sizeof(double));
        B = (double *) calloc(2*n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = B[i] = 1.0;
                A[i+n] = B[i+n] = 1.0/(i+1);
            }
            for (int i = n-m; i < n; i++)
                A[i] = A[i+n] = B[i] = B[i+n] = 0.0;
            kernel1_sph_hi2lo(RP, m, A);
            kernel1_sph_hi2lo(RP, m, A+n);
            kernel2_sph_lo2hi(RP, m, A, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            kernel2_sph_hi2lo(RP, m, A, A+n);
            kernel1_sph_lo2hi(RP, m, A);
            kernel1_sph_lo2hi(RP, m, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            kernel2_sph_lo2hi(RP, m, A, A+n);
            kernel2_sph_hi2lo(RP, m, A, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %17.16e.\n", sqrt(nrm));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = B[i] = 1.0;
                A[i+n] = B[i+n] = 1.0/(i+1);
            }
            for (int i = n-m; i < n; i++)
                A[i] = A[i+n] = B[i] = B[i+n] = 0.0;
            kernel2_sph_hi2lo(RP, m, A, A+n);
            kernel2x4_sph_lo2hi(RP, m, A, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            kernel2x4_sph_hi2lo(RP, m, A, A+n);
            kernel2_sph_lo2hi(RP, m, A, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            kernel2x4_sph_hi2lo(RP, m, A, A+n);
            kernel2x4_sph_lo2hi(RP, m, A, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %17.16e.\n", sqrt(nrm));

        free(A);
        free(B);
        free(RP);
    }

    printf("\n\nTesting the computation of the triangular harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        RP = plan_rottriangle(n, 0.0, -0.5, -0.5);

        printf("\t\tThe 2-norm relative error in sqrt(s^2+c^2): %17.16e.\n", rotnorm(RP)/sqrt(n*(n+1)/2));

        A = (double *) calloc(n, sizeof(double));
        B = (double *) calloc(n, sizeof(double));

        nrm = 0.0;
        for (int m = 1; m < n; m++) {
            for (int i = 0; i < n-m; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-m; i < n; i++)
                A[i] = B[i] = 0.0;
            kernel1_tri_hi2lo(RP, m, A);
            kernel1_tri_lo2hi(RP, m, A);
            nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %17.16e.\n", sqrt(nrm));

        free(A);
        free(B);
        free(RP);
    }

    return 0;
}
