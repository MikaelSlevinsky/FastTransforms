#include "fasttransforms.h"
#include "utilities.h"

const int N = 257;

int main(void) {
    double * A, * Ac, * B;
    RotationPlan * RP;
    SpinRotationPlan * SRP;
    double nrm;

    printf("\n\nTesting the computation of the spherical harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        RP = plan_rotsphere(n);

        printf("\t\tThe 2-norm relative error in sqrt(s^2+c^2): %1.2e.\n", rotnorm(RP)/sqrt(n*(n+1)/2));

        A = (double *) calloc(n, sizeof(double));
        B = (double *) calloc(n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-m; i < n; i++)
                A[i] = B[i] = 0.0;
            kernel_sph_hi2lo(RP, m, A);
            kernel_sph_lo2hi(RP, m, A);
            nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(B);

        A = (double *) calloc(2*n, sizeof(double));
        Ac = (double *) calloc(2*n, sizeof(double));
        B = (double *) calloc(2*n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = Ac[i] = B[i] = 1.0;
                A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
            }
            for (int i = n-m; i < n; i++)
                A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
            kernel_sph_hi2lo(RP, m, A);
            kernel_sph_hi2lo(RP, m, A+n);
            permute(A, Ac, n, 2, 2);
            kernel_sph_lo2hi_SSE(RP, m, Ac);
            permute_t(A, Ac, n, 2, 2);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            permute(A, Ac, n, 2, 2);
            kernel_sph_hi2lo_SSE(RP, m, Ac);
            permute_t(A, Ac, n, 2, 2);
            kernel_sph_lo2hi(RP, m, A);
            kernel_sph_lo2hi(RP, m, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(Ac);
        free(B);

        A = (double *) calloc(4*n, sizeof(double));
        Ac = (double *) calloc(4*n, sizeof(double));
        B = (double *) calloc(4*n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = Ac[i] = B[i] = 1.0;
                A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
            }
            for (int i = n-m; i < n; i++)
                A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = 0.0;
            kernel_sph_hi2lo(RP, m, A);
            kernel_sph_hi2lo(RP, m, A+n);
            kernel_sph_hi2lo(RP, m+2, A+2*n);
            kernel_sph_hi2lo(RP, m+2, A+3*n);
            permute(A, Ac, n, 4, 4);
            kernel_sph_lo2hi_AVX(RP, m, Ac);
            permute_t(A, Ac, n, 4, 4);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            permute(A, Ac, n, 4, 4);
            kernel_sph_hi2lo_AVX(RP, m, Ac);
            permute_t(A, Ac, n, 4, 4);
            kernel_sph_lo2hi(RP, m, A);
            kernel_sph_lo2hi(RP, m, A+n);
            kernel_sph_lo2hi(RP, m+2, A+2*n);
            kernel_sph_lo2hi(RP, m+2, A+3*n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(Ac);
        free(B);
        freeRotationPlan(RP);
    }

    printf("\n\nTesting the computation of the triangular harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        RP = plan_rottriangle(n, 0.0, -0.5, -0.5);

        printf("\t\tThe 2-norm relative error in sqrt(s^2+c^2): %1.2e.\n", rotnorm(RP)/sqrt(n*(n+1)/2));

        A = (double *) calloc(n, sizeof(double));
        B = (double *) calloc(n, sizeof(double));

        nrm = 0.0;
        for (int m = 1; m < n; m++) {
            for (int i = 0; i < n-m; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-m; i < n; i++)
                A[i] = B[i] = 0.0;
            kernel_tri_hi2lo(RP, m, A);
            kernel_tri_lo2hi(RP, m, A);
            nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(B);

        A = (double *) calloc(2*n, sizeof(double));
        Ac = (double *) calloc(2*n, sizeof(double));
        B = (double *) calloc(2*n, sizeof(double));

        nrm = 0.0;
        for (int m = 1; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = Ac[i] = B[i] = 1.0;
                A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
            }
            for (int i = n-m; i < n; i++)
                A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
            kernel_tri_hi2lo(RP, m, A);
            kernel_tri_hi2lo(RP, m+1, A+n);
            permute(A, Ac, n, 2, 2);
            kernel_tri_lo2hi_SSE(RP, m, Ac);
            permute_t(A, Ac, n, 2, 2);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            permute(A, Ac, n, 2, 2);
            kernel_tri_hi2lo_SSE(RP, m, Ac);
            permute_t(A, Ac, n, 2, 2);
            kernel_tri_lo2hi(RP, m, A);
            kernel_tri_lo2hi(RP, m+1, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(Ac);
        free(B);

        A = (double *) calloc(4*n, sizeof(double));
        Ac = (double *) calloc(4*n, sizeof(double));
        B = (double *) calloc(4*n, sizeof(double));

        nrm = 0.0;
        for (int m = 1; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = Ac[i] = B[i] = 1.0;
                A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
            }
            for (int i = n-m; i < n; i++)
                A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = 0.0;
            kernel_tri_hi2lo(RP, m, A);
            kernel_tri_hi2lo(RP, m+1, A+n);
            kernel_tri_hi2lo(RP, m+2, A+2*n);
            kernel_tri_hi2lo(RP, m+3, A+3*n);
            permute(A, Ac, n, 4, 4);
            kernel_tri_lo2hi_AVX(RP, m, Ac);
            permute_t(A, Ac, n, 4, 4);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            permute(A, Ac, n, 4, 4);
            kernel_tri_hi2lo_AVX(RP, m, Ac);
            permute_t(A, Ac, n, 4, 4);
            kernel_tri_lo2hi(RP, m, A);
            kernel_tri_lo2hi(RP, m+1, A+n);
            kernel_tri_lo2hi(RP, m+2, A+2*n);
            kernel_tri_lo2hi(RP, m+3, A+3*n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(Ac);
        free(B);
        freeRotationPlan(RP);
    }

    printf("\n\nTesting the computation of the disk harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        RP = plan_rotdisk(n);

        A = (double *) calloc(n, sizeof(double));
        B = (double *) calloc(n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < 2*n-1; m++) {
            for (int i = 0; i < n-(m+1)/2; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-(m+1)/2; i < n; i++)
                A[i] = B[i] = 0.0;
            kernel_disk_hi2lo(RP, m, A);
            kernel_disk_lo2hi(RP, m, A);
            nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(B);

        A = (double *) calloc(2*n, sizeof(double));
        Ac = (double *) calloc(2*n, sizeof(double));
        B = (double *) calloc(2*n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < 2*n-1; m++) {
            for (int i = 0; i < n-(m+1)/2; i++) {
                A[i] = Ac[i] = B[i] = 1.0;
                A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
            }
            for (int i = n-(m+1)/2; i < n; i++)
                A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
            kernel_disk_hi2lo(RP, m, A);
            kernel_disk_hi2lo(RP, m, A+n);
            permute(A, Ac, n, 2, 2);
            kernel_disk_lo2hi_SSE(RP, m, Ac);
            permute_t(A, Ac, n, 2, 2);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
            permute(A, Ac, n, 2, 2);
            kernel_disk_hi2lo_SSE(RP, m, Ac);
            permute_t(A, Ac, n, 2, 2);
            kernel_disk_lo2hi(RP, m, A);
            kernel_disk_lo2hi(RP, m, A+n);
            nrm += pow(vecnorm_2arg(A, B, n, 2)/vecnorm_1arg(B, n, 2), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

        free(A);
        free(Ac);
        free(B);
        freeRotationPlan(RP);
    }

    printf("\n\nTesting the computation of the spin-weighted spherical harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        for (int s = 0; s < 9; s++) {
            printf("\t\tSpin: %d. ", s);
            RP = plan_rotsphere(n);
            SRP = plan_rotspinsphere(n, s);

            A = (double *) calloc(n, sizeof(double));
            B = (double *) calloc(n, sizeof(double));

            nrm = 0.0;
            for (int m = 0; m < n; m++) {
                for (int i = 0; i < n-MAX(m, s); i++)
                    A[i] = B[i] = 1.0;
                for (int i = n-MAX(m, s); i < n; i++)
                    A[i] = B[i] = 0.0;
                kernel_spinsph_hi2lo(SRP, m, A);
                kernel_spinsph_lo2hi(SRP, m, A);
                nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
                if (s == 0) {
                    kernel_spinsph_hi2lo(SRP, m, A);
                    kernel_sph_lo2hi(RP, m, A);
                    nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
                    kernel_sph_hi2lo(RP, m, A);
                    kernel_spinsph_lo2hi(SRP, m, A);
                    nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
                }
            }
            if (s == 0) nrm /= 3.0;
            printf("The 2-norm relative error in the rotations: %1.2e.\n", sqrt(nrm));

            free(A);
            free(B);
            freeRotationPlan(RP);
            freeSpinRotationPlan(SRP);
        }
    }

    return 0;
}
