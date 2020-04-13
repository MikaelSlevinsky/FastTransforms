#include "ftutilities.h"

int main(void) {
    //struct timeval start, end;

    static double * A;
    static double * B;

    char * FMT = "%3.0f";

    int N, M;//, NTIMES;

    printf("\n\nTesting permute_sph\n\n");

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = 2*N-1;

        A = calloc(N * M, sizeof(double));
        B = calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        permute_sph(A, B, N, M, 2);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = calloc(N * M, sizeof(double));

        permute_t_sph(A, B, N, M, 2);

        printmat("A", FMT, A, N, M);

        permute_sph(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = calloc(N * M, sizeof(double));

        permute_t_sph(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);

        free(A);
        free(B);
    }

    printf("\n\nTesting Warp\n\n");

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = 2*N-1;

        A = calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        printmat("A", FMT, A, N, M);

        warp(A, N, M, 1);

        printmat("A", FMT, A, N, M);

        warp_t(A, N, M, 1);

        warp(A, N, M, 2);

        printmat("A", FMT, A, N, M);

        warp_t(A, N, M, 2);

        warp(A, N, M, 4);

        printmat("A", FMT, A, N, M);

        warp_t(A, N, M, 4);

        free(A);
    }

    printf("\n\nTesting permute_tri\n\n");

    for (int i = 0; i < 1; i++) {
        N = 8*pow(2, i);
        M = N;

        A = calloc(N * M, sizeof(double));
        B = calloc(N * M, sizeof(double));
        for (int i = 0; i < N * M; i++)
            A[i] = (double) i;

        permute_tri(A, B, N, M, 2);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = calloc(N * M, sizeof(double));

        permute_t_tri(A, B, N, M, 2);

        printmat("A", FMT, A, N, M);

        permute_tri(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);
        printmat("B", FMT, B, N, M);

        free(A);

        A = calloc(N * M, sizeof(double));

        permute_t_tri(A, B, N, M, 4);

        printmat("A", FMT, A, N, M);

        free(A);
        free(B);
    }

    return 0;
}
