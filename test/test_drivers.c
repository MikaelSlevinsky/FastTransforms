#include "fasttransforms.h"
#include "ftutilities.h"

double * aligned_copymat(double * A, int n, int m);

int main(int argc, const char * argv[]) {
    int checksum = 0;
    double err = 0.0;
    struct timeval start, end;

    static double * A;
    static double * Ac;
    static double * B;
    ft_rotation_plan * RP;
    ft_rotation_plan * RP1;
    ft_rotation_plan * RP2;
    ft_spin_rotation_plan * SRP;
    ft_harmonic_plan * P;
    ft_spin_harmonic_plan * SP;
    ft_tetrahedral_harmonic_plan * TP;
    //double alpha = -0.5, beta = -0.5, gamma = -0.5, delta = -0.5; // best case scenario
    double alpha = 0.0, beta = 0.0, gamma = 0.0, delta = 0.0; // not as good. perhaps better to transform to second kind Chebyshev

    int IERR, ITIME, J, N, L, M, NTIMES;


    if (argc > 1) {
        sscanf(argv[1], "%d", &IERR);
        if (argc > 2) {
            sscanf(argv[2], "%d", &ITIME);
            if (argc > 3) sscanf(argv[3], "%d", &J);
            else J = 0;
        }
        else ITIME = 1;
    }
    else IERR = 1;

    printf("\nTesting the accuracy of spherical harmonic drivers.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        execute_sph_hi2lo_default(RP, A, M);
        execute_sph_lo2hi_default(RP, A, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        #if defined(__i386__) || defined(__x86_64__)
            execute_sph_hi2lo_SSE2(RP, A, Ac, M);
            execute_sph_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_default(RP, A, M);
            execute_sph_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_AVX(RP, A, Ac, M);
            execute_sph_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_SSE2(RP, A, Ac, M);
            execute_sph_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_sph_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_AVX(RP, A, Ac, M);
            execute_sph_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_AVX512F(RP, A, Ac, M);
            execute_sph_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX512F-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX512F-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_sph_lo2hi_AVX512F(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #elif defined(__aarch64__)
            execute_sph_hi2lo_NEON(RP, A, Ac, M);
            execute_sph_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sph_hi2lo_default(RP, A, M);
            execute_sph_lo2hi_NEON(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #endif

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTiming spherical harmonic drivers.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        FT_TIME(execute_sph_hi2lo_default(RP, A, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(execute_sph_lo2hi_default(RP, A, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        #if defined(__i386__) || defined(__x86_64__)
            FT_TIME(execute_sph_hi2lo_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_lo2hi_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_hi2lo_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_lo2hi_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_hi2lo_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_lo2hi_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_hi2lo_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_lo2hi_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #elif defined(__aarch64__)
            FT_TIME(execute_sph_hi2lo_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sph_lo2hi_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #endif

        printf("\n");
        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical harmonic transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);

        ft_execute_sph2fourier(P, A, N, M);
        ft_execute_fourier2sph(P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }

    printf("\nTiming spherical harmonic transforms.\n\n");
    printf("t2 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);

        FT_TIME(ft_execute_sph2fourier(P, A, N, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(ft_execute_fourier2sph(P, A, N, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical vector field drivers.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        execute_sphv_hi2lo_default(RP, A, M);
        execute_sphv_lo2hi_default(RP, A, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        #if defined(__i386__) || defined(__x86_64__)
            execute_sphv_hi2lo_SSE2(RP, A, Ac, M);
            execute_sphv_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_default(RP, A, M);
            execute_sphv_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_AVX(RP, A, Ac, M);
            execute_sphv_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_SSE2(RP, A, Ac, M);
            execute_sphv_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_sphv_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_AVX(RP, A, Ac, M);
            execute_sphv_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_AVX512F(RP, A, Ac, M);
            execute_sphv_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX512F-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX512F-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_AVX(RP, A, Ac, M);
            execute_sphv_lo2hi_AVX512F(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #elif defined(__aarch64__)
            execute_sphv_hi2lo_NEON(RP, A, Ac, M);
            execute_sphv_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_sphv_hi2lo_default(RP, A, M);
            execute_sphv_lo2hi_NEON(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #endif

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTiming spherical vector field drivers.\n\n");
    printf("t3 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        FT_TIME(execute_sphv_hi2lo_default(RP, A, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(execute_sphv_lo2hi_default(RP, A, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        #if defined(__i386__) || defined(__x86_64__)
            FT_TIME(execute_sphv_hi2lo_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_lo2hi_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_hi2lo_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_lo2hi_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_hi2lo_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_lo2hi_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_hi2lo_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_lo2hi_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #elif defined(__aarch64__)
            FT_TIME(execute_sphv_hi2lo_NEON(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_sphv_lo2hi_NEON(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #endif

        printf("\n");
        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical vector field transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);

        ft_execute_sphv2fourier(P, A, N, M);
        ft_execute_fourier2sphv(P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }

    printf("\nTiming spherical vector field transforms.\n\n");
    printf("t4 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);

        FT_TIME(ft_execute_sphv2fourier(P, A, N, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(ft_execute_fourier2sphv(P, A, N, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Proriol² drivers.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = triones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rottriangle(N, alpha, beta, gamma);

        execute_tri_hi2lo_default(RP, A, M);
        execute_tri_lo2hi_default(RP, A, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 8*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        #if defined(__i386__) || defined(__x86_64__)
            execute_tri_hi2lo_SSE2(RP, A, Ac, M);
            execute_tri_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_default(RP, A, M);
            execute_tri_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_AVX(RP, A, Ac, M);
            execute_tri_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_SSE2(RP, A, Ac, M);
            execute_tri_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_tri_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_AVX(RP, A, Ac, M);
            execute_tri_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_AVX512F(RP, A, Ac, M);
            execute_tri_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX512F-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX512F-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_tri_lo2hi_AVX512F(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #elif defined(__aarch64__)
            execute_tri_hi2lo_NEON(RP, A, Ac, M);
            execute_tri_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_tri_hi2lo_default(RP, A, M);
            execute_tri_lo2hi_NEON(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 8*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #endif

        free(A);
        free(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTiming Proriol² drivers.\n\n");
    printf("t5 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NTIMES = 1 + pow(2048/N, 2);

        A = triones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rottriangle(N, alpha, beta, gamma);

        FT_TIME(execute_tri_hi2lo_default(RP, A, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tri_lo2hi_default(RP, A, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        #if defined(__i386__) || defined(__x86_64__)
            FT_TIME(execute_tri_hi2lo_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_lo2hi_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_hi2lo_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_lo2hi_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_hi2lo_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_lo2hi_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_hi2lo_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_lo2hi_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #elif defined(__aarch64__)
            FT_TIME(execute_tri_hi2lo_NEON(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_tri_lo2hi_NEON(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #endif

        printf("\n");
        free(A);
        free(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Proriol² transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = trirand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);

        ft_execute_tri2cheb(P, A, N, M);
        ft_execute_cheb2tri(P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 8*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }

    printf("\nTiming Proriol² transforms.\n\n");
    printf("t6 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NTIMES = 1 + pow(2048/N, 2);

        A = trirand(N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);

        FT_TIME(ft_execute_tri2cheb(P, A, N, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(ft_execute_cheb2tri(P, A, N, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Zernike drivers.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rotdisk(N);

        execute_disk_hi2lo_default(RP, A, M);
        execute_disk_lo2hi_default(RP, A, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ default-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        #if defined(__i386__) || defined(__x86_64__)
            execute_disk_hi2lo_SSE2(RP, A, Ac, M);
            execute_disk_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_default(RP, A, M);
            execute_disk_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-SSE2 \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_AVX(RP, A, Ac, M);
            execute_disk_lo2hi_SSE2(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-SSE2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_SSE2(RP, A, Ac, M);
            execute_disk_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ SSE2-AVX \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_disk_lo2hi_AVX(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_AVX(RP, A, Ac, M);
            execute_disk_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_AVX512F(RP, A, Ac, M);
            execute_disk_lo2hi_AVX_FMA(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX512F-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX512F-AVX_FMA \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_AVX_FMA(RP, A, Ac, M);
            execute_disk_lo2hi_AVX512F(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 AVX_FMA-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ AVX_FMA-AVX512F \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #elif defined(__aarch64__)
            execute_disk_hi2lo_NEON(RP, A, Ac, M);
            execute_disk_lo2hi_default(RP, A, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ NEON-default \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);

            execute_disk_hi2lo_default(RP, A, M);
            execute_disk_lo2hi_NEON(RP, A, Ac, M);

            err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
            printf("ϵ_2 default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
            printf("ϵ_∞ default-NEON \t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
            ft_checktest(err, 2*N, &checksum);
        #endif

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }

    printf("\nTiming Zernike drivers.\n\n");
    printf("t7 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NTIMES = 1 + pow(2048/N, 2);

        A = diskones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rotdisk(N);

        FT_TIME(execute_disk_hi2lo_default(RP, A, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(execute_disk_lo2hi_default(RP, A, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        #if defined(__i386__) || defined(__x86_64__)
            FT_TIME(execute_disk_hi2lo_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_lo2hi_SSE2(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_hi2lo_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_lo2hi_AVX(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_hi2lo_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_lo2hi_AVX_FMA(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_hi2lo_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_lo2hi_AVX512F(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #elif defined(__aarch64__)
            FT_TIME(execute_disk_hi2lo_NEON(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            FT_TIME(execute_disk_lo2hi_NEON(RP, A, B, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));
        #endif

        printf("\n");
        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Zernike transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_disk2cxf(N);

        ft_execute_disk2cxf(P, A, N, M);
        ft_execute_cxf2disk(P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 8*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }

    printf("\nTiming Zernike transforms.\n\n");
    printf("t8 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NTIMES = 1 + pow(2048/N, 2);

        A = diskrand(N, M);
        P = ft_plan_disk2cxf(N);

        FT_TIME(ft_execute_disk2cxf(P, A, N, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(ft_execute_cxf2disk(P, A, N, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Proriol³ drivers.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;

        A = tetones(N, L, M);
        Ac = aligned_copymat(A, N, L*M);
        B = copymat(A, N, L*M);

        RP1 = ft_plan_rottriangle(N, alpha, beta, gamma + delta + 1.0);
        RP2 = ft_plan_rottriangle(N, beta, gamma, delta);

        ft_execute_tet_hi2lo(RP1, RP2, A, L, M);
        ft_execute_tet_lo2hi(RP1, RP2, A, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 default \t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ default \t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        execute_tet_hi2lo_SSE2(RP1, RP2, A, Ac, L, M);
        ft_execute_tet_lo2hi(RP1, RP2, A, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 SSE2-default (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ SSE2-default (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        ft_execute_tet_hi2lo(RP1, RP2, A, L, M);
        execute_tet_lo2hi_SSE2(RP1, RP2, A, Ac, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 default-SSE2 (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ default-SSE2 (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        execute_tet_hi2lo_AVX(RP1, RP2, A, Ac, L, M);
        execute_tet_lo2hi_SSE2(RP1, RP2, A, Ac, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 AVX-SSE2 \t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ AVX-SSE2 \t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        execute_tet_hi2lo_SSE2(RP1, RP2, A, Ac, L, M);
        execute_tet_lo2hi_AVX(RP1, RP2, A, Ac, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 SSE2-AVX \t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ SSE2-AVX \t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        execute_tet_hi2lo_AVX512F(RP1, RP2, A, Ac, L, M);
        execute_tet_lo2hi_AVX(RP1, RP2, A, Ac, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 AVX512F-AVX  (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ AVX512F-AVX  (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        execute_tet_hi2lo_AVX(RP1, RP2, A, Ac, L, M);
        execute_tet_lo2hi_AVX512F(RP1, RP2, A, Ac, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 AVX-AVX512F  (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ AVX-AVX512F  (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP1);
        ft_destroy_rotation_plan(RP2);
    }

    printf("\nTiming Proriol³ drivers.\n\n");
    printf("t9 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;
        NTIMES = 1 + pow(512/N, 2);

        A = tetones(N, L, M);
        B = aligned_copymat(A, N, L*M);

        RP1 = ft_plan_rottriangle(N, alpha, beta, gamma + delta + 1.0);
        RP2 = ft_plan_rottriangle(N, beta, gamma, delta);

        FT_TIME(ft_execute_tet_hi2lo(RP1, RP2, A, L, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(ft_execute_tet_lo2hi(RP1, RP2, A, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tet_hi2lo_SSE2(RP1, RP2, A, B, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tet_lo2hi_SSE2(RP1, RP2, A, B, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tet_hi2lo_AVX(RP1, RP2, A, B, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tet_lo2hi_AVX(RP1, RP2, A, B, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tet_hi2lo_AVX512F(RP1, RP2, A, B, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        FT_TIME(execute_tet_lo2hi_AVX512F(RP1, RP2, A, B, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP1);
        ft_destroy_rotation_plan(RP2);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Proriol³ transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;

        A = tetrand(N, L, M);
        B = copymat(A, N, L*M);
        TP = ft_plan_tet2cheb(N, alpha, beta, gamma, delta);

        ft_execute_tet2cheb(TP, A, N, L, M);
        ft_execute_cheb2tet(TP, A, N, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 \t\t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ \t\t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        free(A);
        free(B);
        ft_destroy_tetrahedral_harmonic_plan(TP);
    }

    printf("\nTiming Proriol³ transforms.\n\n");
    printf("t10 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;
        NTIMES = 1 + pow(256/N, 2);

        A = tetrand(N, L, M);
        TP = ft_plan_tet2cheb(N, alpha, beta, gamma, delta);

        FT_TIME(ft_execute_tet2cheb(TP, A, N, L, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME(ft_execute_cheb2tet(TP, A, N, L, M), start, end, NTIMES)
        printf("  %.6f", elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        ft_destroy_tetrahedral_harmonic_plan(TP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spin-weighted spherical harmonic drivers.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        for (int S = -4; S <= 4; S++) {
            ft_complex * AC = spinsphones(N, M, S);
            A = (double *) AC;
            ft_complex * AcC = (ft_complex *) aligned_copymat(A, 2*N, M);
            ft_complex * BC = (ft_complex *) copymat(A, 2*N, M);
            Ac = (double *) AcC;
            B = (double *) BC;
            SRP = ft_plan_rotspinsphere(N, S);

            execute_spinsph_hi2lo_default(SRP, AC, M);
            execute_spinsph_lo2hi_default(SRP, AC, M);

            err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
            printf("ϵ_2 default-default \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
            printf("ϵ_∞ default-default \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
            ft_checktest(err, 2*N, &checksum);

            #if defined(__i386__) || defined(__x86_64__)
                execute_spinsph_hi2lo_SSE2(SRP, AC, M);
                execute_spinsph_lo2hi_default(SRP, AC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 SSE2-default \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ SSE2-default \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);

                execute_spinsph_hi2lo_default(SRP, AC, M);
                execute_spinsph_lo2hi_SSE2(SRP, AC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 default-SSE2 \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ default-SSE2 \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);

                execute_spinsph_hi2lo_AVX(SRP, AC, AcC, M);
                execute_spinsph_lo2hi_SSE2(SRP, AC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 AVX-SSE2 \t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ AVX-SSE2 \t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);

                execute_spinsph_hi2lo_SSE2(SRP, AC, M);
                execute_spinsph_lo2hi_AVX(SRP, AC, AcC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 SSE2-AVX \t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ SSE2-AVX \t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);

                execute_spinsph_hi2lo_AVX_FMA(SRP, AC, AcC, M);
                execute_spinsph_lo2hi_AVX(SRP, AC, AcC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 AVX_FMA-AVX \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ AVX_FMA-AVX \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);

                execute_spinsph_hi2lo_AVX(SRP, AC, AcC, M);
                execute_spinsph_lo2hi_AVX_FMA(SRP, AC, AcC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 AVX-AVX_FMA \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ AVX-AVX_FMA \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);
            #elif defined(__aarch64__)
                execute_spinsph_hi2lo_NEON(SRP, AC, M);
                execute_spinsph_lo2hi_default(SRP, AC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 NEON-default \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ NEON-default \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);

                execute_spinsph_hi2lo_default(SRP, AC, M);
                execute_spinsph_lo2hi_NEON(SRP, AC, M);

                err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
                printf("ϵ_2 default-NEON \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 4*sqrt(N), &checksum);
                err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
                printf("ϵ_∞ default-NEON \t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
                ft_checktest(err, 2*N, &checksum);
            #endif

            free(AC);
            VFREE(AcC);
            free(BC);
            ft_destroy_spin_rotation_plan(SRP);
        }
    }

    printf("\nTiming spin-weighted spherical harmonic drivers.\n\n");
    printf("t11 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        printf("%d\n", N);

        for (int S = -2; S <= 2; S++) {
            ft_complex * AC = spinsphones(N, M, S);
            ft_complex * BC = (ft_complex *) copymat((double *) AC, 2*N, M);
            SRP = ft_plan_rotspinsphere(N, S);

            FT_TIME(execute_spinsph_hi2lo_default(SRP, AC, M), start, end, NTIMES)
            printf("%d  %.6f", S, elapsed(&start, &end, NTIMES));

            FT_TIME(execute_spinsph_lo2hi_default(SRP, AC, M), start, end, NTIMES)
            printf("  %.6f", elapsed(&start, &end, NTIMES));

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(execute_spinsph_hi2lo_SSE2(SRP, AC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));

                FT_TIME(execute_spinsph_lo2hi_SSE2(SRP, AC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));

                FT_TIME(execute_spinsph_hi2lo_AVX(SRP, AC, BC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));

                FT_TIME(execute_spinsph_lo2hi_AVX(SRP, AC, BC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));

                FT_TIME(execute_spinsph_hi2lo_AVX_FMA(SRP, AC, BC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));

                FT_TIME(execute_spinsph_lo2hi_AVX_FMA(SRP, AC, BC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(execute_spinsph_hi2lo_NEON(SRP, AC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));

                FT_TIME(execute_spinsph_lo2hi_NEON(SRP, AC, M), start, end, NTIMES)
                printf("  %.6f", elapsed(&start, &end, NTIMES));
            #endif

            free(AC);
            free(BC);
            ft_destroy_spin_rotation_plan(SRP);
        }
        printf("\n");
    }
    printf("];\n");

    printf("\nTesting the accuracy of spin-weighted spherical harmonic transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        for (int S = -4; S <= 4; S++) {
            ft_complex * AC = spinsphrand(N, M, S);
            ft_complex * BC = (ft_complex *) copymat((double *) AC, 2*N, M);
            A = (double *) AC;
            B = (double *) BC;
            SP = ft_plan_spinsph2fourier(N, S);

            ft_execute_spinsph2fourier(SP, AC, N, M);
            ft_execute_fourier2spinsph(SP, AC, N, M);

            err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
            printf("ϵ_2 \t\t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
            printf("ϵ_∞ \t\t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
            ft_checktest(err, 2*N, &checksum);

            free(AC);
            free(BC);
            ft_destroy_spin_harmonic_plan(SP);
        }
    }

    printf("\nTiming spin-weighted spherical harmonic transforms.\n\n");
    printf("t12 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        printf("%d\n", N);
        for (int S = -2; S <= 2; S++) {
            ft_complex * AC = spinsphrand(N, M, S);
            SP = ft_plan_spinsph2fourier(N, S);

            FT_TIME(ft_execute_spinsph2fourier(SP, AC, N, M), start, end, NTIMES)
            printf("%d  %.6f", S, elapsed(&start, &end, NTIMES));

            FT_TIME(ft_execute_fourier2spinsph(SP, AC, N, M), start, end, NTIMES)
            printf("  %.6f\n", elapsed(&start, &end, NTIMES));

            free(AC);
            ft_destroy_spin_harmonic_plan(SP);
        }
    }
    printf("];\n");

    return checksum;
}

#define A(i,j) A[(i)+n*(j)]

double * aligned_copymat(double * A, int n, int m) {
    double * B = VMALLOC(VALIGN(n)*m*sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            B[(i)+VALIGN(n)*(j)] = A(i,j);
    for (int i = n; i < VALIGN(n); i++)
        for (int j = 0; j < m; j++)
            B[(i)+VALIGN(n)*(j)] = 0.0;
    return B;
}
