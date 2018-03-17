#include "../src/drivers.h"
#include "utilities.h"

int main(void) {
    struct timeval start, end;
    double delta;

    static double * A;
    static double * B;
    RotationPlan * RP;
    SphericalHarmonicPlan * P;

    int N, M, NLOOPS;

    printf("err1 = [\n");
    for (int i = 0; i < 6; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;

        A = sphones(N, M);
        B = sphones(N, M);
        RP = plan_rotsphere(N);

        execute_sph_hi2lo(RP, A, M);
        execute_sph_lo2hi(RP, A, M);
        printf("%17.16e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%17.16e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));
        //printf("%17.16e\n", vecnorm_2arg(A, B, N, M)/sqrt(N*(N+1.0)/2.0));

        free(A);
        free(B);
        free(RP);
    }
    printf("];\n");

    printf("t1 = [\n");
    for (int i = 0; i < 6; i++) {
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
        printf("  %.6f\n", delta/NLOOPS);

        free(A);
        free(B);
        free(RP);
    }
    printf("];\n");


    printf("err2 = [\n");
    for (int i = 0; i < 6; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;

        A = sphrand(N, M);
        B = copyA(A, N, M);
        P = plan_sph2fourier(N);

        execute_sph2fourier(P, A, N, M);
        execute_fourier2sph(P, A, N, M);

        printf("%17.16e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%17.16e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));
        //printf("%17.16e\n", vecnorm_2arg(A, B, N, M)/sqrt(N*(N+1.0)/2.0));

        free(A);
        free(B);
        free(P);
    }
    printf("];\n");

    printf("t2 = [\n");
    for (int i = 0; i < 6; i++) {
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

    return 0;
}
