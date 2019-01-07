#include "ftutilities.h"

int main(void) {
    struct timeval start, end;

    static double * A;
    static double * B;

    char * FMT = "%3.0f";

    int N, M, NLOOPS;

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = 2*N-1;

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        permute_sph(A, B, N, M, 2);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = (double *) calloc(N * M, sizeof(double));

        permute_t_sph(A, B, N, M, 2);

        printmat("A", FMT, A, N, M);

        permute_sph(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = (double *) calloc(N * M, sizeof(double));

        permute_t_sph(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);

        free(A);
        free(B);

        M = 2*N+1;

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        permute_sph(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = (double *) calloc(N * M, sizeof(double));

        permute_t_sph(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);

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
            permute_sph_SSE(A, B, N, M);
            permute_t_sph_SSE(A, B, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        free(A);
        free(B);

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

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

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        free(A);
        free(B);

        A = (double *) calloc(N * M, sizeof(double));
        B = (double *) calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

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
