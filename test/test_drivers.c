#include "fasttransforms.h"
#include "ftutilities.h"

double * aligned_copymat(double * A, int n, int m);

int main(int argc, const char * argv[]) {
    struct timeval start, end;

    static double * A;
    static double * Ac;
    static double * B;
    ft_rotation_plan * RP;
    ft_rotation_plan * RP1;
    ft_rotation_plan * RP2;
    ft_spin_rotation_plan * SRP;
    ft_harmonic_plan * P;
    ft_tetrahedral_harmonic_plan * TP;
    //double alpha = -0.5, beta = -0.5, gamma = -0.5, delta = -0.5; // best case scenario
    double alpha = 0.0, beta = 0.0, gamma = 0.0, delta = 0.0; // not as good. perhaps better to transform to second kind Chebyshev

    int IERR, ITIME, J, N, L, M, NLOOPS;


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
    printf("err1 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        ft_execute_sph_hi2lo(RP, A, M);
        ft_execute_sph_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sph_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_sph_lo2hi(RP, A, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sph_hi2lo(RP, A, M);
        ft_execute_sph_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sph_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_sph_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sph_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_sph_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sph_hi2lo_AVX512(RP, A, Ac, M);
        ft_execute_sph_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sph_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_sph_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic drivers.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);
        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical harmonic transforms.\n\n");
    printf("err2 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);

        ft_execute_sph2fourier(P, A, N, M);
        ft_execute_fourier2sph(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic transforms.\n\n");
    printf("t2 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph2fourier(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_fourier2sph(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical vector field drivers.\n\n");
    printf("err3 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        ft_execute_sphv_hi2lo(RP, A, M);
        ft_execute_sphv_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sphv_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_sphv_lo2hi(RP, A, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sphv_hi2lo(RP, A, M);
        ft_execute_sphv_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sphv_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_sphv_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sphv_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_sphv_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sphv_hi2lo_AVX512(RP, A, Ac, M);
        ft_execute_sphv_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_sphv_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_sphv_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTiming spherical vector field drivers.\n\n");
    printf("t3 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rotsphere(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);
        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical vector field transforms.\n\n");
    printf("err4 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);

        ft_execute_sphv2fourier(P, A, N, M);
        ft_execute_fourier2sphv(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTiming spherical vector field transforms.\n\n");
    printf("t4 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv2fourier(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_fourier2sphv(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of triangular harmonic drivers.\n\n");
    printf("err5 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = triones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rottriangle(N, alpha, beta, gamma);

        ft_execute_tri_hi2lo(RP, A, M);
        ft_execute_tri_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_tri_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_tri_lo2hi(RP, A, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_tri_hi2lo(RP, A, M);
        ft_execute_tri_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_tri_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_tri_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_tri_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_tri_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_tri_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_tri_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_tri_hi2lo_AVX512(RP, A, Ac, M);
        ft_execute_tri_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTiming triangular harmonic drivers.\n\n");
    printf("t5 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NLOOPS = 1 + pow(2048/N, 2);

        A = triones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rottriangle(N, alpha, beta, gamma);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of triangular harmonic transforms.\n\n");
    printf("err6 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = trirand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);

        ft_execute_tri2cheb(P, A, N, M);
        ft_execute_cheb2tri(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTiming triangular harmonic transforms.\n\n");
    printf("t6 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NLOOPS = 1 + pow(2048/N, 2);

        A = trirand(N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri2cheb(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_cheb2tri(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of disk harmonic drivers.\n\n");
    printf("err7 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskones(N, M);
        Ac = aligned_copymat(A, N, M);
        B = copymat(A, N, M);
        RP = ft_plan_rotdisk(N);

        ft_execute_disk_hi2lo(RP, A, M);
        ft_execute_disk_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_disk_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_disk_lo2hi(RP, A, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_disk_hi2lo(RP, A, M);
        ft_execute_disk_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_disk_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_disk_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_disk_hi2lo_SSE(RP, A, Ac, M);
        ft_execute_disk_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_disk_hi2lo_AVX512(RP, A, Ac, M);
        ft_execute_disk_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        ft_execute_disk_hi2lo_AVX(RP, A, Ac, M);
        ft_execute_disk_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTiming disk harmonic drivers.\n\n");
    printf("t7 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NLOOPS = 1 + pow(2048/N, 2);

        A = diskones(N, M);
        B = aligned_copymat(A, N, M);
        RP = ft_plan_rotdisk(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of disk harmonic transforms.\n\n");
    printf("err8 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_disk2cxf(N);

        ft_execute_disk2cxf(P, A, N, M);
        ft_execute_cxf2disk(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTiming disk harmonic transforms.\n\n");
    printf("t8 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NLOOPS = 1 + pow(2048/N, 2);

        A = diskrand(N, M);
        P = ft_plan_disk2cxf(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk2cxf(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_cxf2disk(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of tetrahedral harmonic drivers.\n\n");
    printf("err9 = [\n");
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

        printf("%d  %1.2e  ", N, norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        ft_execute_tet_hi2lo_SSE(RP1, RP2, A, Ac, L, M);
        ft_execute_tet_lo2hi(RP1, RP2, A, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        ft_execute_tet_hi2lo(RP1, RP2, A, L, M);
        ft_execute_tet_lo2hi_SSE(RP1, RP2, A, Ac, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        ft_execute_tet_hi2lo_AVX(RP1, RP2, A, Ac, L, M);
        ft_execute_tet_lo2hi_SSE(RP1, RP2, A, Ac, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        ft_execute_tet_hi2lo_SSE(RP1, RP2, A, Ac, L, M);
        ft_execute_tet_lo2hi_AVX(RP1, RP2, A, Ac, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        ft_execute_tet_hi2lo_AVX512(RP1, RP2, A, Ac, L, M);
        ft_execute_tet_lo2hi_AVX(RP1, RP2, A, Ac, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e  ", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        ft_execute_tet_hi2lo_AVX(RP1, RP2, A, Ac, L, M);
        ft_execute_tet_lo2hi_AVX512(RP1, RP2, A, Ac, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        free(A);
        VFREE(Ac);
        free(B);
        ft_destroy_rotation_plan(RP1);
        ft_destroy_rotation_plan(RP2);
    }
    printf("];\n");

    printf("\nTiming tetrahedral harmonic drivers.\n\n");
    printf("t9 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;
        NLOOPS = 1 + pow(512/N, 2);

        A = tetones(N, L, M);
        B = aligned_copymat(A, N, L*M);

        RP1 = ft_plan_rottriangle(N, alpha, beta, gamma + delta + 1.0);
        RP2 = ft_plan_rottriangle(N, beta, gamma, delta);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_hi2lo(RP1, RP2, A, L, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_lo2hi(RP1, RP2, A, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_hi2lo_SSE(RP1, RP2, A, B, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_lo2hi_SSE(RP1, RP2, A, B, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_hi2lo_AVX(RP1, RP2, A, B, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_lo2hi_AVX(RP1, RP2, A, B, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_hi2lo_AVX512(RP1, RP2, A, B, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet_lo2hi_AVX512(RP1, RP2, A, B, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        ft_destroy_rotation_plan(RP1);
        ft_destroy_rotation_plan(RP2);
    }
    printf("];\n");

    printf("\nTesting the accuracy of tetrahedral harmonic transforms.\n\n");
    printf("err10 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;

        A = tetrand(N, L, M);
        B = copymat(A, N, L*M);
        TP = ft_plan_tet2cheb(N, alpha, beta, gamma, delta);

        ft_execute_tet2cheb(TP, A, N, L, M);
        ft_execute_cheb2tet(TP, A, N, L, M);

        printf("%1.2e  ", norm_2arg(A, B, N*L*M)/norm_1arg(B, N*L*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*L*M)/normInf_1arg(B, N*L*M));

        free(A);
        free(B);
        ft_destroy_tetrahedral_harmonic_plan(TP);
    }
    printf("];\n");

    printf("\nTiming tetrahedral harmonic transforms.\n\n");
    printf("t10 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;
        NLOOPS = 1 + pow(512/N, 2);

        A = tetrand(N, L, M);
        TP = ft_plan_tet2cheb(N, alpha, beta, gamma, delta);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tet2cheb(TP, A, N, L, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_cheb2tet(TP, A, N, L, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_tetrahedral_harmonic_plan(TP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spin-weighted spherical harmonic drivers.\n\n");
    printf("err11 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        printf("%d", N);
        for (int S = 0; S < 9; S++) {
            A = spinsphones(N, M, S);
            Ac = aligned_copymat(A, N, M);
            B = copymat(A, N, M);
            SRP = ft_plan_rotspinsphere(N, S);

            ft_execute_spinsph_hi2lo(SRP, A, M);
            ft_execute_spinsph_lo2hi(SRP, A, M);

            printf("  %1.2e", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
            printf("  %1.2e", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

            ft_execute_spinsph_hi2lo_SSE(SRP, A, Ac, M);
            ft_execute_spinsph_lo2hi_SSE(SRP, A, Ac, M);

            printf("  %1.2e", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
            printf("  %1.2e", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

            ft_execute_spinsph_hi2lo(SRP, A, M);
            ft_execute_spinsph_lo2hi_SSE(SRP, A, Ac, M);

            printf("  %1.2e", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
            printf("  %1.2e", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

            free(A);
            VFREE(Ac);
            free(B);
            ft_destroy_spin_rotation_plan(SRP);
        }
        printf("\n");
    }
    printf("];\n");

    printf("\nTiming spin-weighted spherical harmonic drivers.\n\n");
    printf("t11 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        printf("%d", N);

        for (int S = 0; S < 9; S++) {
            A = spinsphones(N, M, S);
            B = copymat(A, N, M);
            SRP = ft_plan_rotspinsphere(N, S);

            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
                ft_execute_spinsph_hi2lo(SRP, A, M);
            }
            gettimeofday(&end, NULL);

            printf("  %.6f", elapsed(&start, &end, NLOOPS));

            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
                ft_execute_spinsph_lo2hi(SRP, A, M);
            }
            gettimeofday(&end, NULL);

            printf("  %.6f", elapsed(&start, &end, NLOOPS));

            free(A);
            free(B);
            ft_destroy_spin_rotation_plan(SRP);
        }
        printf("\n");
    }
    printf("];\n");

    return 0;
}

#define A(i,j) A[(i)+n*(j)]

double * aligned_copymat(double * A, int n, int m) {
    double * B = (double *) VMALLOC(VALIGN(n)*m*sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            B[(i)+VALIGN(n)*(j)] = A(i,j);
    for (int i = n; i < VALIGN(n); i++)
        for (int j = 0; j < m; j++)
            B[(i)+VALIGN(n)*(j)] = 0.0;
    return B;
}
