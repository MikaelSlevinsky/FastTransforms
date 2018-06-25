#include "fasttransforms.h"
#include "utilities.h"

int main(void) {
    struct timeval start, end;
    double delta;

    static double * A;
    static double * Ac;
    static double * B;
    RotationPlan * RP;
    SphericalHarmonicPlan * P;

    int N, M, NLOOPS;

    printf("err1 = [\n");
    for (int i = 0; i < 3; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;

        A = sphones(N, M);
        Ac = copyA(A, N, M);
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

        execute_sph_hi2lo_SSE(RP, A, Ac, M);
        execute_sph_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(Ac);
        free(B);
        free(RP);

        M = 2*N+1;

        A = sphones(N, M);
        Ac = copyA(A, N, M);
        B = copyA(A, N, M);
        RP = plan_rotsphere(N);

        execute_sph_hi2lo_SSE(RP, A, Ac, M);
        execute_sph_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(Ac);
        free(B);
        free(RP);
    }
    printf("];\n");

    printf("t1 = [\n");
    for (int i = 3; i <6; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;
        NLOOPS = 1 + pow(4096/N, 2);

        A = sphones(N, M);
        B = sphones(N, M);
        RP = plan_rotsphere(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d  %.6f", N, delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", delta/NLOOPS);

        free(A);
        free(B);
        free(RP);

        M = 2*N+1;
        NLOOPS = 1 + pow(4096/N, 2);

        A = sphones(N, M);
        B = sphones(N, M);
        RP = plan_rotsphere(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf(" %.6f", delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf(" %.6f\n", delta/NLOOPS);

        free(A);
        free(B);
        free(RP);
    }
    printf("];\n");

/*
    printf("err2 = [\n");
    for (int i = 0; i < 8; i++) {
        N = 64*pow(2, i);
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
        free(P);
    }
    printf("];\n");

    printf("t2 = [\n");
    for (int i = 0; i < 8; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;
        NLOOPS = 1 + pow(4096/N, 2);

        A = sphrand(N, M);
        P = plan_sph2fourier(N);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph2fourier(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d  %.6f", N, delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_fourier2sph(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f\n", delta/NLOOPS);

        free(A);
        free(P);
    }
    printf("];\n");

    //double alpha = -0.5, beta = -0.5, gamma = -0.5; // best case scenario
    double alpha = 0.0, beta = 0.0, gamma = 0.0; // not as good. perhaps better to transform to second kind Chebyshev

    TriangularHarmonicPlan * Q;

    printf("err3 = [\n");
    for (int i = 0; i < 6; i++) {
        N = 64*pow(2, i);
        M = N;

        A = triones(N, M);
        Ac = copyA(A, N, M);
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

        execute_tri_hi2lo_SSE(RP, A, Ac, M);
        execute_tri_lo2hi_SSE(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_AVX(RP, A, Ac, M);
        execute_tri_lo2hi(RP, A, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo(RP, A, M);
        execute_tri_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e  ", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        execute_tri_hi2lo_AVX(RP, A, Ac, M);
        execute_tri_lo2hi_AVX(RP, A, Ac, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(Ac);
        free(B);
        free(RP);
    }
    printf("];\n");

    printf("t3 = [\n");
    for (int i = 0; i < 6; i++) {
        N = 64*pow(2, i);
        M = N;
        NLOOPS = 1 + pow(4096/N, 2);

        A = triones(N, M);
        B = triones(N, M);
        RP = plan_rottriangle(N, alpha, beta, gamma);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo(RP, A, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d  %.6f", N, delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi(RP, A, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", N, delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi_SSE(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_hi2lo_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", N, delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_lo2hi_AVX(RP, A, B, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f\n", delta/NLOOPS);

        free(A);
        free(B);
        free(RP);
    }
    printf("];\n");


    printf("err4 = [\n");
    for (int i = 0; i < 8; i++) {
        N = 64*pow(2, i);
        M = N;

        A = trirand(N, M);
        B = copyA(A, N, M);
        Q = plan_tri2cheb(N, alpha, beta, gamma);

        execute_tri2cheb(Q, A, N, M);
        execute_cheb2tri(Q, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        free(Q);
    }
    printf("];\n");

    printf("t4 = [\n");
    for (int i = 0; i < 8; i++) {
        N = 64*pow(2, i);
        M = N;
        NLOOPS = 1 + pow(4096/N, 2);

        A = trirand(N, M);
        Q = plan_tri2cheb(N, alpha, beta, gamma);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri2cheb(Q, A, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d  %.6f", N, delta/NLOOPS);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_cheb2tri(Q, A, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f\n", delta/NLOOPS);

        free(A);
        free(Q);
    }
    printf("];\n");
*/
    return 0;
}
