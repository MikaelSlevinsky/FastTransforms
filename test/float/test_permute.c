#include "fasttransformsf.h"
#include "utilitiesf.h"

int main(void) {
    struct timeval start, end;

    static float * A;
    static float * B;

    int N, M, NLOOPS;

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = 2*N-1;

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        permute_sph(A, B, N, M, 2);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (float *) calloc(N * M, sizeof(float));

        permute_t_sph(A, B, N, M, 2);

        printmat("A", A, N, M);

        permute_sph(A, B, N, M, 4);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (float *) calloc(N * M, sizeof(float));

        permute_t_sph(A, B, N, M, 4);

        printmat("A", A, N, M);

        free(A);
        free(B);

        M = 2*N+1;

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        permute_sph(A, B, N, M, 4);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (float *) calloc(N * M, sizeof(float));

        permute_t_sph(A, B, N, M, 4);

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

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            permute_sph_SSE(A, B, N, M);
            permute_t_sph_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        free(A);
        free(B);

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            permute_sph_AVX(A, B, N, M);
            permute_t_sph_AVX(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        free(B);
    }
    printf("];\n");

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = N;

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        permute_tri_SSE(A, B, N, M);

        printmat("A", A, N, M);
        printmat("B", B, N, M);

        free(A);

        A = (float *) calloc(N * M, sizeof(float));

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

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            permute_tri_SSE(A, B, N, M);
            permute_t_tri_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        free(A);
        free(B);

        A = (float *) calloc(N * M, sizeof(float));
        B = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N * M; i++)
            A[i] = (float) i;

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            permute_tri_AVX(A, B, N, M);
            permute_t_tri_AVX(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        free(B);
    }
    printf("];\n");
*/
    return 0;
}
