#include "fasttransforms.h"
#include "utilities.h"

int main(int argc, const char * argv[]) {
    struct timeval start, end;

    static double * A;
    static double * Ac;
    static double * B;
    RotationPlan * RP;
    SpinRotationPlan * SRP;
    HarmonicPlan * P;
    //double alpha = -0.5, beta = -0.5, gamma = -0.5; // best case scenario
    double alpha = 0.0, beta = 0.0, gamma = 0.0; // not as good. perhaps better to transform to second kind Chebyshev

    int IERR, ITIME, J, N, M, NLOOPS;


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
        Ac = copyAlign(A, N, M);
        B = copyA(A, N, M);
        RP = plan_rotsphere(N);

        execute_sph_hi2lo(RP, A, M);
        execute_sph_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_sph_hi2lo_SSE(RP, A, Ac, M);
        execute_sph_lo2hi(RP, A, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_sph_hi2lo(RP, A, M);
        execute_sph_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_sph_hi2lo_AVX(RP, A, Ac, M);
        execute_sph_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_sph_hi2lo_SSE(RP, A, Ac, M);
        execute_sph_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_sph_hi2lo_AVX512(RP, A, Ac, M);
        execute_sph_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_sph_hi2lo_AVX(RP, A, Ac, M);
        execute_sph_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        VFREE(Ac);
        free(B);
        freeRotationPlan(RP);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic drivers.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphones(N, M);
        B = copyAlign(A, N, M);
        RP = plan_rotsphere(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);
        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        freeRotationPlan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical harmonic transforms.\n\n");
    printf("err2 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copyA(A, N, M);
        P = plan_sph2fourier(N);

        execute_sph2fourier(P, A, N, M);
        execute_fourier2sph(P, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        freeHarmonicPlan(P);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic transforms.\n\n");
    printf("t2 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = plan_sph2fourier(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph2fourier(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_fourier2sph(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        freeHarmonicPlan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of triangular harmonic drivers.\n\n");
    printf("err3 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = triones(N, M);
        Ac = copyAlign(A, N, M);
        B = copyA(A, N, M);
        RP = plan_rottriangle(N, alpha, beta, gamma);

        execute_tri_hi2lo(RP, A, M);
        execute_tri_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_SSE(RP, A, Ac, M);
        execute_tri_lo2hi(RP, A, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo(RP, A, M);
        execute_tri_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_AVX(RP, A, Ac, M);
        execute_tri_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_SSE(RP, A, Ac, M);
        execute_tri_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_AVX(RP, A, Ac, M);
        execute_tri_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_AVX512(RP, A, Ac, M);
        execute_tri_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        VFREE(Ac);
        free(B);
        freeRotationPlan(RP);
    }
    printf("];\n");


    printf("\nTiming triangular harmonic drivers.\n\n");
    printf("t3 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NLOOPS = 1 + pow(2048/N, 2);

        A = triones(N, M);
        B = copyAlign(A, N, M);
        RP = plan_rottriangle(N, alpha, beta, gamma);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        freeRotationPlan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of triangular harmonic transforms.\n\n");
    printf("err4 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = trirand(N, M);
        B = copyA(A, N, M);
        P = plan_tri2cheb(N, alpha, beta, gamma);

        execute_tri2cheb(P, A, N, M);
        execute_cheb2tri(P, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        freeHarmonicPlan(P);
    }
    printf("];\n");

    printf("\nTiming triangular harmonic transforms.\n\n");
    printf("t4 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NLOOPS = 1 + pow(2048/N, 2);

        A = trirand(N, M);
        P = plan_tri2cheb(N, alpha, beta, gamma);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri2cheb(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_cheb2tri(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        freeHarmonicPlan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of disk harmonic drivers.\n\n");
    printf("err5 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskones(N, M);
        Ac = copyAlign(A, N, M);
        B = copyA(A, N, M);
        RP = plan_rotdisk(N);

        execute_disk_hi2lo(RP, A, M);
        execute_disk_lo2hi(RP, A, M);

        printf("%d  %1.2e  ", N, vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_disk_hi2lo_SSE(RP, A, Ac, M);
        execute_disk_lo2hi(RP, A, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_disk_hi2lo(RP, A, M);
        execute_disk_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_disk_hi2lo_AVX(RP, A, Ac, M);
        execute_disk_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_disk_hi2lo_SSE(RP, A, Ac, M);
        execute_disk_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_disk_hi2lo_AVX512(RP, A, Ac, M);
        execute_disk_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_disk_hi2lo_AVX(RP, A, Ac, M);
        execute_disk_lo2hi_AVX512(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        VFREE(Ac);
        free(B);
        freeRotationPlan(RP);
    }
    printf("];\n");

    printf("\nTiming disk harmonic drivers.\n\n");
    printf("t5 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NLOOPS = 1 + pow(2048/N, 2);

        A = diskones(N, M);
        B = copyAlign(A, N, M);
        RP = plan_rotdisk(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_hi2lo_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f", elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_lo2hi_AVX512(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        VFREE(B);
        freeRotationPlan(RP);
    }
    printf("];\n");

    printf("\nTesting the accuracy of disk harmonic transforms.\n\n");
    printf("err6 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskrand(N, M);
        B = copyA(A, N, M);
        P = plan_disk2cxf(N);

        execute_disk2cxf(P, A, N, M);
        execute_cxf2disk(P, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        freeHarmonicPlan(P);
    }
    printf("];\n");

    printf("\nTiming disk harmonic transforms.\n\n");
    printf("t6 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NLOOPS = 1 + pow(2048/N, 2);

        A = diskrand(N, M);
        P = plan_disk2cxf(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk2cxf(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_cxf2disk(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        freeHarmonicPlan(P);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spin-weighted spherical harmonic drivers.\n\n");
    printf("err7 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        printf("%d", N);
        for (int S = 0; S < 9; S++) {
            A = spinsphones(N, M, S);
            Ac = copyAlign(A, N, M);
            B = copyA(A, N, M);
            SRP = plan_rotspinsphere(N, S);

            execute_spinsph_hi2lo(SRP, A, M);
            execute_spinsph_lo2hi(SRP, A, M);

            printf("  %1.2e", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
            printf("  %1.2e", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

            execute_spinsph_hi2lo_SSE(SRP, A, Ac, M);
            execute_spinsph_lo2hi_SSE(SRP, A, Ac, M);

            printf("  %1.2e", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
            printf("  %1.2e", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

            execute_spinsph_hi2lo(SRP, A, M);
            execute_spinsph_lo2hi_SSE(SRP, A, Ac, M);

            printf("  %1.2e", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
            printf("  %1.2e", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

            free(A);
            VFREE(Ac);
            free(B);
            freeSpinRotationPlan(SRP);
        }
        printf("\n");
    }
    printf("];\n");

    printf("\nTiming spin-weighted spherical harmonic drivers.\n\n");
    printf("t7 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        printf("%d", N);

        for (int S = 0; S < 9; S++) {
            A = spinsphones(N, M, S);
            B = copyA(A, N, M);
            SRP = plan_rotspinsphere(N, S);

            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
                execute_spinsph_hi2lo(SRP, A, M);
            }
            gettimeofday(&end, NULL);

            printf("  %.6f", elapsed(&start, &end, NLOOPS));

            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
                execute_spinsph_lo2hi(SRP, A, M);
            }
            gettimeofday(&end, NULL);

            printf("  %.6f", elapsed(&start, &end, NLOOPS));

            free(A);
            free(B);
            freeSpinRotationPlan(SRP);
        }
        printf("\n");
    }
    printf("];\n");

    return 0;
}
