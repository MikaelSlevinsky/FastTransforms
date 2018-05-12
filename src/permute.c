// Local data transposes to enable SSE and AVX vectorization.

#include "fasttransforms.h"

void local_transpose_SSE(const double * A, double * B, const int N, const int M) {
    for (int i = 0; i < N; i++)
        B[i] = A[i];
    for (int j = 1; j < M; j += 2)
        for (int i = 0; i < 2*N; i++)
            B[(2*i)%(2*N)+(2*i)/(2*N)+j*N] = A[i+j*N];
}

void local_reverse_transpose_SSE(double * A, const double * B, const int N, const int M) {
    for (int i = 0; i < N; i++)
        A[i] = B[i];
    for (int j = 1; j < M; j += 2)
        for (int i = 0; i < 2*N; i++)
            A[i+j*N] = B[(2*i)%(2*N)+(2*i)/(2*N)+j*N];
}

void local_even_transpose_SSE(const double * B, double * D, const int N, const int M) {
    for (int i = 0; i < N; i++)
        D[i] = B[i];
    for (int j = 1; j < 1+2*(M/4); j += 2)
        for (int i = 0; i < 2*N; i++)
            D[(2*i)%(2*N)+(2*i)/(2*N)+j*N] = B[i+j*N];
}

void local_even_reverse_transpose_SSE(double * B, const double * D, const int N, const int M) {
    for (int i = 0; i < N; i++)
        B[i] = D[i];
    for (int j = 1; j < 1+2*(M/4); j += 2)
        for (int i = 0; i < 2*N; i++)
            B[i+j*N] = D[(2*i)%(2*N)+(2*i)/(2*N)+j*N];
}

void local_odd_transpose_SSE(const double * C, double * E, const int N, const int M) {
    for (int j = 0; j < 2*((M+1)/4); j += 2)
        for (int i = 0; i < 2*N; i++)
            E[(2*i)%(2*N)+(2*i)/(2*N)+j*N] = C[i+j*N];
}

void local_odd_reverse_transpose_SSE(double * C, const double * E, const int N, const int M) {
    for (int j = 0; j < 2*((M+1)/4); j += 2)
        for (int i = 0; i < 2*N; i++)
            C[i+j*N] = E[(2*i)%(2*N)+(2*i)/(2*N)+j*N];
}


void permute_tri_SSE(const double * A, double * B, const int N, const int M) {
    for (int j = 0; j < M; j += 2)
        for (int i = 0; i < 2*N; i++)
            B[(2*i)%(2*N)+(2*i)/(2*N)+j*N] = A[i+j*N];
}

void permute_t_tri_SSE(double * A, const double * B, const int N, const int M) {
    for (int j = 0; j < M; j += 2)
        for (int i = 0; i < 2*N; i++)
            A[i+j*N] = B[(2*i)%(2*N)+(2*i)/(2*N)+j*N];
}

void permute_tri_AVX(const double * A, double * B, const int N, const int M) {
    for (int j = 0; j < M; j += 4)
        for (int i = 0; i < 4*N; i++)
            B[(4*i)%(4*N)+(4*i)/(4*N)+j*N] = A[i+j*N];
}

void permute_t_tri_AVX(double * A, const double * B, const int N, const int M) {
    for (int j = 0; j < M; j += 4)
        for (int i = 0; i < 4*N; i++)
            A[i+j*N] = B[(4*i)%(4*N)+(4*i)/(4*N)+j*N];
}
