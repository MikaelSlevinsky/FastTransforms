#include "fasttransforms.h"
#include "utilities.h"

int main(void) {
    struct timeval start, end;
    double delta;

    static double * A;
    static double * B;

    int N, M, NLOOPS;

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = 2*N-1;

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        local_transpose_SSE(A, B, N, M);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (double *) calloc(N * M, sizeof(double));

        local_reverse_transpose_SSE(A, B, N, M);

        printmat("A", A, N, M);

        free(A);
        free(B);

        A = (double *) calloc(N * (1+2*(M/4)), sizeof(double));
        B = (double *) calloc(N * (1+2*(M/4)), sizeof(double));
        for (int i = 0; i < N * (1+2*(M/4)); i++)
            A[i] = (double) i;

        local_even_transpose_SSE(A, B, N, M);

        printmat("A", A, N, 1 + 2*(M/4));
        printmat("B", B, N, 1 + 2*(M/4));

        free(A);

        A = (double *) calloc(N * (1+2*(M/4)), sizeof(double));

        local_even_reverse_transpose_SSE(A, B, N, M);

        printmat("A", A, N, 1 + 2*(M/4));

        free(A);
        free(B);

        A = (double *) calloc(N * (2*((M+1)/4)), sizeof(double));
        B = (double *) calloc(N * (2*((M+1)/4)), sizeof(double));
        for (int i = 0; i < N * (2*((M+1)/4)); i++)
            A[i] = (double) i;

        local_odd_transpose_SSE(A, B, N, M);

        printmat("A", A, N, 2*((M+1)/4));
        printmat("B", B, N, 2*((M+1)/4));

        free(A);

        A = (double *) calloc(N * (2*((M+1)/4)), sizeof(double));

        local_odd_reverse_transpose_SSE(A, B, N, M);

        printmat("A", A, N, 2*((M+1)/4));

        free(A);
        free(B);

        M = 2*N+1;

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        permute_sph_AVX(A, B, N, M);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (double *) calloc(N * M, sizeof(double));

        permute_t_sph_AVX(A, B, N, M);

        printmat("A", A, N, M);

        free(A);
        free(B);
    }
/*
    printf("t1 = [\n");
    for (int i = 0; i < 8; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;
        NLOOPS = 1 + pow(8192/N, 2);

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            local_transpose_SSE(A, B, N, M);
            local_reverse_transpose_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d  %.6f", N, delta/NLOOPS);

        free(A);
        free(B);

        A = (double *) calloc(N * (1+2*(M/4)), sizeof(double));
        B = (double *) calloc(N * (1+2*(M/4)), sizeof(double));
        for (int i = 0; i < N * (1+2*(M/4)); i++)
            A[i] = (double) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            local_even_transpose_SSE(A, B, N, M);
            local_even_reverse_transpose_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f", N, delta/NLOOPS);

        free(A);
        free(B);

        A = (double *) calloc(N * (2*((M+1)/4)), sizeof(double));
        B = (double *) calloc(N * (2*((M+1)/4)), sizeof(double));
        for (int i = 0; i < N * (2*((M+1)/4)); i++)
            A[i] = (double) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            local_odd_transpose_SSE(A, B, N, M);
            local_odd_reverse_transpose_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("  %.6f\n", delta/NLOOPS);

        free(A);
        free(B);
    }
    printf("];\n");

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = N;

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        permute_tri_SSE(A, B, N, M);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (double *) calloc(N * M, sizeof(double));

        permute_t_tri_SSE(A, B, N, M);

        printmat("A", A, N, M);

        free(A);
        free(B);
    }

    printf("t2 = [\n");
    for (int i = 0; i < 8; i++) {
        N = 64*pow(2, i);
        M = N;
        NLOOPS = 1 + pow(8192/N, 2);

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            permute_tri_SSE(A, B, N, M);
            permute_t_tri_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        printf("%d  %.6f\n", N, delta/NLOOPS);

        free(A);
        free(B);
    }
    printf("];\n");
*/
    return 0;
}
