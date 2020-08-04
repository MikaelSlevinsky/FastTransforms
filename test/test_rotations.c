#include "fasttransforms.h"
#include "ftutilities.h"

double rotnorm(const ft_rotation_plan * RP);

const int N = 257;

int main(void) {
    int checksum = 0;
    double err;
    double * A, * Ac, * B;
    ft_rotation_plan * RP;
    ft_spin_rotation_plan * SRP;

    printf("\nTesting the computation of the spherical harmonic Givens rotations.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        RP = ft_plan_rotsphere(n);
        err = rotnorm(RP)/sqrt(n*(n+1)/2);
        printf("Departure from sqrt(s^2+c^2)-1 at \t     n = %3i: \t |%20.2e ", n, err);
        ft_checktest(err, 1, &checksum);

        err = 0;
        A = calloc(n, sizeof(double));
        B = calloc(n, sizeof(double));
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-m; i < n; i++)
                A[i] = B[i] = 0.0;
            ft_kernel_sph_hi2lo(RP, m%2, m, A, 1);
            ft_kernel_sph_lo2hi(RP, m%2, m, A, 1);
            err += pow(ft_norm_2arg(A, B, n)/ft_norm_1arg(B, n), 2);
        }
        err = sqrt(err);
        printf("Applying the rotations with one   column  at n = %3i: \t |%20.2e ", n, err);
        ft_checktest(err, 2*n, &checksum);
        free(A);
        free(B);

        #if defined(__i386__) || defined(__x86_64__)
            err = 0;
            A = calloc(2*n, sizeof(double));
            Ac = VMALLOC(2*n*sizeof(double));
            B = calloc(2*n, sizeof(double));
            for (int m = 2; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
                kernel_sph_hi2lo_default(RP, m%2, m, A, 1);
                kernel_sph_hi2lo_default(RP, m%2, m, A+n, 1);
                permute(A, Ac, n, 2, 2);
                kernel_sph_lo2hi_SSE2(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                permute(A, Ac, n, 2, 2);
                kernel_sph_hi2lo_SSE2(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                kernel_sph_lo2hi_default(RP, m%2, m, A, 1);
                kernel_sph_lo2hi_default(RP, m%2, m, A+n, 1);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with two   columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);

            err = 0;
            A = calloc(4*n, sizeof(double));
            Ac = VMALLOC(4*n*sizeof(double));
            B = calloc(4*n, sizeof(double));
            for (int m = 2; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                    A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                    A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = 0.0;
                kernel_sph_hi2lo_default(RP, m%2, m, A, 1);
                kernel_sph_hi2lo_default(RP, m%2, m, A+n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+2, A+2*n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+2, A+3*n, 1);
                permute(A, Ac, n, 4, 4);
                kernel_sph_lo2hi_AVX_FMA(RP, m%2, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
                permute(A, Ac, n, 4, 4);
                kernel_sph_hi2lo_AVX(RP, m%2, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                kernel_sph_lo2hi_default(RP, m%2, m, A, 1);
                kernel_sph_lo2hi_default(RP, m%2, m, A+n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+2, A+2*n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+2, A+3*n, 1);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
                permute(A, Ac, n, 4, 4);
                kernel_sph_hi2lo_AVX_FMA(RP, m%2, m, Ac, 4);
                kernel_sph_lo2hi_AVX(RP, m%2, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with four  columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);

            err = 0;
            A = calloc(8*n, sizeof(double));
            Ac = VMALLOC(8*n*sizeof(double));
            B = calloc(8*n, sizeof(double));
            for (int m = 2; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                    A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                    A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
                    A[i+4*n] = Ac[i+4*n] = B[i+4*n] = 1.0/pow(i+1, 4);
                    A[i+5*n] = Ac[i+5*n] = B[i+5*n] = 1.0/pow(i+1, 5);
                    A[i+6*n] = Ac[i+6*n] = B[i+6*n] = 1.0/pow(i+1, 6);
                    A[i+7*n] = Ac[i+7*n] = B[i+7*n] = 1.0/pow(i+1, 7);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = A[i+4*n] = A[i+5*n] = A[i+6*n] = A[i+7*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = Ac[i+4*n] = Ac[i+5*n] = Ac[i+6*n] = Ac[i+7*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = B[i+4*n] = B[i+5*n] = B[i+6*n] = B[i+7*n] = 0.0;
                kernel_sph_hi2lo_default(RP, m%2, m, A, 1);
                kernel_sph_hi2lo_default(RP, m%2, m, A+n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+2, A+2*n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+2, A+3*n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+4, A+4*n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+4, A+5*n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+6, A+6*n, 1);
                kernel_sph_hi2lo_default(RP, m%2, m+6, A+7*n, 1);
                permute(A, Ac, n, 8, 8);
                kernel_sph_lo2hi_AVX512F(RP, m%2, m, Ac, 8);
                permute_t(A, Ac, n, 8, 8);
                err += pow(ft_norm_2arg(A, B, 8*n)/ft_norm_1arg(B, 8*n), 2);
                permute(A, Ac, n, 8, 8);
                kernel_sph_hi2lo_AVX512F(RP, m%2, m, Ac, 8);
                permute_t(A, Ac, n, 8, 8);
                kernel_sph_lo2hi_default(RP, m%2, m, A, 1);
                kernel_sph_lo2hi_default(RP, m%2, m, A+n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+2, A+2*n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+2, A+3*n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+4, A+4*n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+4, A+5*n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+6, A+6*n, 1);
                kernel_sph_lo2hi_default(RP, m%2, m+6, A+7*n, 1);
                err += pow(ft_norm_2arg(A, B, 8*n)/ft_norm_1arg(B, 8*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with eight columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);
        #elif defined(__aarch64__)
            err = 0;
            A = calloc(2*n, sizeof(double));
            Ac = VMALLOC(2*n*sizeof(double));
            B = calloc(2*n, sizeof(double));
            for (int m = 2; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
                kernel_sph_hi2lo_default(RP, m%2, m, A, 1);
                kernel_sph_hi2lo_default(RP, m%2, m, A+n, 1);
                permute(A, Ac, n, 2, 2);
                kernel_sph_lo2hi_NEON(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                permute(A, Ac, n, 2, 2);
                kernel_sph_hi2lo_NEON(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                kernel_sph_lo2hi_default(RP, m%2, m, A, 1);
                kernel_sph_lo2hi_default(RP, m%2, m, A+n, 1);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with two   columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);
        #endif
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTesting the computation of the triangular harmonic Givens rotations.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        RP = ft_plan_rottriangle(n, 0.0, -0.5, -0.5);
        err = rotnorm(RP)/sqrt(n*(n+1)/2);
        printf("Departure from sqrt(s^2+c^2)-1 at \t     n = %3i: \t |%20.2e ", n, err);
        ft_checktest(err, 1, &checksum);

        err = 0;
        A = calloc(n, sizeof(double));
        B = calloc(n, sizeof(double));
        for (int m = 1; m < n; m++) {
            for (int i = 0; i < n-m; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-m; i < n; i++)
                A[i] = B[i] = 0.0;
            ft_kernel_tri_hi2lo(RP, 0, m, A, 1);
            ft_kernel_tri_lo2hi(RP, 0, m, A, 1);
            err += pow(ft_norm_2arg(A, B, n)/ft_norm_1arg(B, n), 2);
        }
        err = sqrt(err);
        printf("Applying the rotations with one   column  at n = %3i: \t |%20.2e ", n, err);
        ft_checktest(err, n, &checksum);
        free(A);
        free(B);

        #if defined(__i386__) || defined(__x86_64__)
            err = 0;
            A = calloc(2*n, sizeof(double));
            Ac = VMALLOC(2*n*sizeof(double));
            B = calloc(2*n, sizeof(double));
            for (int m = 1; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
                kernel_tri_hi2lo_default(RP, 0, m, A, 1);
                kernel_tri_hi2lo_default(RP, 0, m+1, A+n, 1);
                permute(A, Ac, n, 2, 2);
                kernel_tri_lo2hi_SSE2(RP, 0, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                permute(A, Ac, n, 2, 2);
                kernel_tri_hi2lo_SSE2(RP, 0, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                kernel_tri_lo2hi_default(RP, 0, m, A, 1);
                kernel_tri_lo2hi_default(RP, 0, m+1, A+n, 1);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with two   columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);

            err = 0;
            A = calloc(4*n, sizeof(double));
            Ac = VMALLOC(4*n*sizeof(double));
            B = calloc(4*n, sizeof(double));
            for (int m = 1; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                    A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                    A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = 0.0;
                kernel_tri_hi2lo_default(RP, 0, m, A, 1);
                kernel_tri_hi2lo_default(RP, 0, m+1, A+n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+2, A+2*n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+3, A+3*n, 1);
                permute(A, Ac, n, 4, 4);
                kernel_tri_lo2hi_AVX_FMA(RP, 0, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
                permute(A, Ac, n, 4, 4);
                kernel_tri_hi2lo_AVX(RP, 0, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                kernel_tri_lo2hi_default(RP, 0, m, A, 1);
                kernel_tri_lo2hi_default(RP, 0, m+1, A+n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+2, A+2*n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+3, A+3*n, 1);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
                permute(A, Ac, n, 4, 4);
                kernel_tri_hi2lo_AVX_FMA(RP, 0, m, Ac, 4);
                kernel_tri_lo2hi_AVX(RP, 0, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with four  columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);

            err = 0;
            A = calloc(8*n, sizeof(double));
            Ac = VMALLOC(8*n*sizeof(double));
            B = calloc(8*n, sizeof(double));
            for (int m = 1; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                    A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                    A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
                    A[i+4*n] = Ac[i+4*n] = B[i+4*n] = 1.0/pow(i+1, 4);
                    A[i+5*n] = Ac[i+5*n] = B[i+5*n] = 1.0/pow(i+1, 5);
                    A[i+6*n] = Ac[i+6*n] = B[i+6*n] = 1.0/pow(i+1, 6);
                    A[i+7*n] = Ac[i+7*n] = B[i+7*n] = 1.0/pow(i+1, 7);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = A[i+4*n] = A[i+5*n] = A[i+6*n] = A[i+7*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = Ac[i+4*n] = Ac[i+5*n] = Ac[i+6*n] = Ac[i+7*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = B[i+4*n] = B[i+5*n] = B[i+6*n] = B[i+7*n] = 0.0;
                kernel_tri_hi2lo_default(RP, 0, m, A, 1);
                kernel_tri_hi2lo_default(RP, 0, m+1, A+n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+2, A+2*n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+3, A+3*n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+4, A+4*n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+5, A+5*n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+6, A+6*n, 1);
                kernel_tri_hi2lo_default(RP, 0, m+7, A+7*n, 1);
                permute(A, Ac, n, 8, 8);
                kernel_tri_lo2hi_AVX512F(RP, 0, m, Ac, 8);
                permute_t(A, Ac, n, 8, 8);
                err += pow(ft_norm_2arg(A, B, 8*n)/ft_norm_1arg(B, 8*n), 2);
                permute(A, Ac, n, 8, 8);
                kernel_tri_hi2lo_AVX512F(RP, 0, m, Ac, 8);
                permute_t(A, Ac, n, 8, 8);
                kernel_tri_lo2hi_default(RP, 0, m, A, 1);
                kernel_tri_lo2hi_default(RP, 0, m+1, A+n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+2, A+2*n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+3, A+3*n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+4, A+4*n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+5, A+5*n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+6, A+6*n, 1);
                kernel_tri_lo2hi_default(RP, 0, m+7, A+7*n, 1);
                err += pow(ft_norm_2arg(A, B, 8*n)/ft_norm_1arg(B, 8*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with eight columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);
        #elif defined(__aarch64__)
            err = 0;
            A = calloc(2*n, sizeof(double));
            Ac = VMALLOC(2*n*sizeof(double));
            B = calloc(2*n, sizeof(double));
            for (int m = 1; m < n; m++) {
                for (int i = 0; i < n-m; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                }
                for (int i = n-m; i < n; i++)
                    A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
                kernel_tri_hi2lo_default(RP, 0, m, A, 1);
                kernel_tri_hi2lo_default(RP, 0, m+1, A+n, 1);
                permute(A, Ac, n, 2, 2);
                kernel_tri_lo2hi_NEON(RP, 0, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                permute(A, Ac, n, 2, 2);
                kernel_tri_hi2lo_NEON(RP, 0, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                kernel_tri_lo2hi_default(RP, 0, m, A, 1);
                kernel_tri_lo2hi_default(RP, 0, m+1, A+n, 1);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with two   columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);
        #endif
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTesting the computation of the disk harmonic Givens rotations.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        RP = ft_plan_rotdisk(n);

        err = 0;
        A = calloc(n, sizeof(double));
        B = calloc(n, sizeof(double));
        for (int m = 2; m < 2*n-1; m++) {
            for (int i = 0; i < n-(m+1)/2; i++)
                A[i] = B[i] = 1.0;
            for (int i = n-(m+1)/2; i < n; i++)
                A[i] = B[i] = 0.0;
            ft_kernel_disk_hi2lo(RP, m%2, m, A, 1);
            ft_kernel_disk_lo2hi(RP, m%2, m, A, 1);
            err += pow(ft_norm_2arg(A, B, n)/ft_norm_1arg(B, n), 2);
        }
        err = sqrt(err);
        printf("Applying the rotations with one   column  at n = %3i: \t |%20.2e ", n, err);
        ft_checktest(err, n, &checksum);
        free(A);
        free(B);

        #if defined(__i386__) || defined(__x86_64__)
            err = 0;
            A = calloc(2*n, sizeof(double));
            Ac = VMALLOC(2*n*sizeof(double));
            B = calloc(2*n, sizeof(double));
            for (int m = 2; m < 2*n-1; m++) {
                for (int i = 0; i < n-(m+1)/2; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                }
                for (int i = n-(m+1)/2; i < n; i++)
                    A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
                kernel_disk_hi2lo_default(RP, m%2, m, A, 1);
                kernel_disk_hi2lo_default(RP, m%2, m, A+n, 1);
                permute(A, Ac, n, 2, 2);
                kernel_disk_lo2hi_SSE2(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                permute(A, Ac, n, 2, 2);
                kernel_disk_hi2lo_SSE2(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                kernel_disk_lo2hi_default(RP, m%2, m, A, 1);
                kernel_disk_lo2hi_default(RP, m%2, m, A+n, 1);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with two   columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);

            err = 0;
            A = calloc(4*n, sizeof(double));
            Ac = VMALLOC(4*n*sizeof(double));
            B = calloc(4*n, sizeof(double));
            for (int m = 2; m < 2*n-1; m++) {
                for (int i = 0; i < n-(m+1)/2; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                    A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                    A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
                }
                for (int i = n-(m+1)/2; i < n; i++)
                    A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = 0.0;
                kernel_disk_hi2lo_default(RP, m%2, m, A, 1);
                kernel_disk_hi2lo_default(RP, m%2, m, A+n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+2, A+2*n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+2, A+3*n, 1);
                permute(A, Ac, n, 4, 4);
                kernel_disk_lo2hi_AVX_FMA(RP, m%2, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
                permute(A, Ac, n, 4, 4);
                kernel_disk_hi2lo_AVX(RP, m%2, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                kernel_disk_lo2hi_default(RP, m%2, m, A, 1);
                kernel_disk_lo2hi_default(RP, m%2, m, A+n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+2, A+2*n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+2, A+3*n, 1);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
                permute(A, Ac, n, 4, 4);
                kernel_disk_hi2lo_AVX_FMA(RP, m%2, m, Ac, 4);
                kernel_disk_lo2hi_AVX(RP, m%2, m, Ac, 4);
                permute_t(A, Ac, n, 4, 4);
                err += pow(ft_norm_2arg(A, B, 4*n)/ft_norm_1arg(B, 4*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with four  columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);

            err = 0;
            A = calloc(8*n, sizeof(double));
            Ac = VMALLOC(8*n*sizeof(double));
            B = calloc(8*n, sizeof(double));
            for (int m = 2; m < 2*n-1; m++) {
                for (int i = 0; i < n-(m+1)/2; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                    A[i+2*n] = Ac[i+2*n] = B[i+2*n] = 1.0/pow(i+1, 2);
                    A[i+3*n] = Ac[i+3*n] = B[i+3*n] = 1.0/pow(i+1, 3);
                    A[i+4*n] = Ac[i+4*n] = B[i+4*n] = 1.0/pow(i+1, 4);
                    A[i+5*n] = Ac[i+5*n] = B[i+5*n] = 1.0/pow(i+1, 5);
                    A[i+6*n] = Ac[i+6*n] = B[i+6*n] = 1.0/pow(i+1, 6);
                    A[i+7*n] = Ac[i+7*n] = B[i+7*n] = 1.0/pow(i+1, 7);
                }
                for (int i = n-(m+1)/2; i < n; i++)
                    A[i] = A[i+n] = A[i+2*n] = A[i+3*n] = A[i+4*n] = A[i+5*n] = A[i+6*n] = A[i+7*n] = Ac[i] = Ac[i+n] = Ac[i+2*n] = Ac[i+3*n] = Ac[i+4*n] = Ac[i+5*n] = Ac[i+6*n] = Ac[i+7*n] = B[i] = B[i+n] = B[i+2*n] = B[i+3*n] = B[i+4*n] = B[i+5*n] = B[i+6*n] = B[i+7*n] = 0.0;
                kernel_disk_hi2lo_default(RP, m%2, m, A, 1);
                kernel_disk_hi2lo_default(RP, m%2, m, A+n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+2, A+2*n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+2, A+3*n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+4, A+4*n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+4, A+5*n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+6, A+6*n, 1);
                kernel_disk_hi2lo_default(RP, m%2, m+6, A+7*n, 1);
                permute(A, Ac, n, 8, 8);
                kernel_disk_lo2hi_AVX512F(RP, m%2, m, Ac, 8);
                permute_t(A, Ac, n, 8, 8);
                err += pow(ft_norm_2arg(A, B, 8*n)/ft_norm_1arg(B, 8*n), 2);
                permute(A, Ac, n, 8, 8);
                kernel_disk_hi2lo_AVX512F(RP, m%2, m, Ac, 8);
                permute_t(A, Ac, n, 8, 8);
                kernel_disk_lo2hi_default(RP, m%2, m, A, 1);
                kernel_disk_lo2hi_default(RP, m%2, m, A+n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+2, A+2*n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+2, A+3*n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+4, A+4*n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+4, A+5*n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+6, A+6*n, 1);
                kernel_disk_lo2hi_default(RP, m%2, m+6, A+7*n, 1);
                err += pow(ft_norm_2arg(A, B, 8*n)/ft_norm_1arg(B, 8*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with eight columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);
        #elif defined(__aarch64__)
            err = 0;
            A = calloc(2*n, sizeof(double));
            Ac = VMALLOC(2*n*sizeof(double));
            B = calloc(2*n, sizeof(double));
            for (int m = 2; m < 2*n-1; m++) {
                for (int i = 0; i < n-(m+1)/2; i++) {
                    A[i] = Ac[i] = B[i] = 1.0;
                    A[i+n] = Ac[i+n] = B[i+n] = 1.0/(i+1);
                }
                for (int i = n-(m+1)/2; i < n; i++)
                    A[i] = A[i+n] = Ac[i] = Ac[i+n] = B[i] = B[i+n] = 0.0;
                kernel_disk_hi2lo_default(RP, m%2, m, A, 1);
                kernel_disk_hi2lo_default(RP, m%2, m, A+n, 1);
                permute(A, Ac, n, 2, 2);
                kernel_disk_lo2hi_NEON(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                permute(A, Ac, n, 2, 2);
                kernel_disk_hi2lo_NEON(RP, m%2, m, Ac, 2);
                permute_t(A, Ac, n, 2, 2);
                kernel_disk_lo2hi_default(RP, m%2, m, A, 1);
                kernel_disk_lo2hi_default(RP, m%2, m, A+n, 1);
                err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
            }
            err = sqrt(err);
            printf("Applying the rotations with two   columns at n = %3i: \t |%20.2e ", n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            VFREE(Ac);
            free(B);
        #endif
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTesting the computation of the spin-weighted spherical harmonic Givens rotations.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        for (int s = -4; s <= 4; s++) {
            int as = abs(s);
            RP = ft_plan_rotsphere(n);
            SRP = ft_plan_rotspinsphere(n, s);

            err = 0;
            A = calloc(2*n, sizeof(double));
            ft_complex * AC = (ft_complex *) A;
            B = calloc(2*n, sizeof(double));
            err = 0;
            for (int m = -n+1; m < n; m++) {
                int am = abs(m);
                for (int i = 0; i < n-MAX(am, as); i++)
                    A[2*i] = A[2*i+1] = B[2*i] = B[2*i+1] = 1.0;
                for (int i = n-MAX(am, as); i < n; i++)
                    A[2*i] = A[2*i+1] = B[2*i] = B[2*i+1] = 0.0;
                #if defined(__i386__) || defined(__x86_64__)
                    kernel_spinsph_hi2lo_default(SRP, m, AC, 1);
                    kernel_spinsph_lo2hi_SSE2(SRP, m, AC, 1);
                    err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                    kernel_spinsph_hi2lo_SSE2(SRP, m, AC, 1);
                    kernel_spinsph_lo2hi_default(SRP, m, AC, 1);
                    err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                #elif defined(__aarch64__)
                    kernel_spinsph_hi2lo_default(SRP, m, AC, 1);
                    kernel_spinsph_lo2hi_NEON(SRP, m, AC, 1);
                    err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                    kernel_spinsph_hi2lo_NEON(SRP, m, AC, 1);
                    kernel_spinsph_lo2hi_default(SRP, m, AC, 1);
                    err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                #endif
                if (s == 0) {
                    kernel_spinsph_hi2lo_default(SRP, m, AC, 1);
                    kernel_sph_lo2hi_default(RP, am%2, am, A, 2);
                    kernel_sph_lo2hi_default(RP, am%2, am, A+1, 2);
                    err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                    kernel_sph_hi2lo_default(RP, am%2, am, A, 2);
                    kernel_sph_hi2lo_default(RP, am%2, am, A+1, 2);
                    kernel_spinsph_lo2hi_default(SRP, m, AC, 1);
                    err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                    #if defined(__i386__) || defined(__x86_64__)
                        kernel_sph_hi2lo_default(RP, am%2, am, A, 2);
                        kernel_sph_hi2lo_default(RP, am%2, am, A+1, 2);
                        kernel_spinsph_lo2hi_SSE2(SRP, m, AC, 1);
                        err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                        kernel_spinsph_hi2lo_SSE2(SRP, m, AC, 1);
                        kernel_sph_lo2hi_default(RP, am%2, am, A, 2);
                        kernel_sph_lo2hi_default(RP, am%2, am, A+1, 2);
                        err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                    #elif defined(__aarch64__)
                        kernel_sph_hi2lo_default(RP, am%2, am, A, 2);
                        kernel_sph_hi2lo_default(RP, am%2, am, A+1, 2);
                        kernel_spinsph_lo2hi_NEON(SRP, m, AC, 1);
                        err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                        kernel_spinsph_hi2lo_NEON(SRP, m, AC, 1);
                        kernel_sph_lo2hi_default(RP, am%2, am, A, 2);
                        kernel_sph_lo2hi_default(RP, am%2, am, A+1, 2);
                        err += pow(ft_norm_2arg(A, B, 2*n)/ft_norm_1arg(B, 2*n), 2);
                    #endif
                }
            }
            if (s == 0) err /= 6.0;
            err = sqrt(err);
            printf("Applying the rotations with spin s = %2i at n = %3i: \t |%20.2e ", s, n, err);
            ft_checktest(err, 2*n, &checksum);
            free(A);
            free(B);
            ft_destroy_rotation_plan(RP);
            ft_destroy_spin_rotation_plan(SRP);
        }
    }
    printf("\n");
    return checksum;
}

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

double rotnorm(const ft_rotation_plan * RP) {
    double * s = RP->s, * c = RP->c;
    double ret = 0.0;
    int n = RP->n;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++)
            ret += pow(hypot(s(l,m), c(l,m)) - 1.0, 2);
    return sqrt(ret);
}
