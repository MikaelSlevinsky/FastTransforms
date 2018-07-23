// Permutations that enable SSE and AVX vectorization.

#include "fasttransformsf.h"

void permute(const float * A, float * B, const int N, const int M, const int L) {
    #pragma omp parallel for if (N < 2*M)
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            B[(L*i)%(L*N)+(L*i)/(L*N)+j*N] = A[i+j*N];
}

void permute_t(float * A, const float * B, const int N, const int M, const int L) {
    #pragma omp parallel for if (N < 2*M)
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            A[i+j*N] = B[(L*i)%(L*N)+(L*i)/(L*N)+j*N];
}


void permute_sph(const float * A, float * B, const int N, const int M, const int L) {
    if (L == 2) {
        for (int i = 0; i < N; i++)
            B[i] = A[i];
        permute(A+N, B+N, N, M-1, 2);
    }
    else {
        permute_sph(A, B, N, M%(2*L), L/2);
        permute(A+(M%(2*L))*N, B+(M%(2*L))*N, N, M-M%(2*L), L);
    }
}

void permute_t_sph(float * A, const float * B, const int N, const int M, const int L) {
    if (L == 2) {
        for (int i = 0; i < N; i++)
            A[i] = B[i];
        permute_t(A+N, B+N, N, M-1, 2);
    }
    else {
        permute_t_sph(A, B, N, M%(2*L), L/2);
        permute_t(A+(M%(2*L))*N, B+(M%(2*L))*N, N, M-M%(2*L), L);
    }
}

void permute_tri(const float * A, float * B, const int N, const int M, const int L) {
    if (L == 2) {
        permute(A, B, N, M, 2);
    }
    else {
        permute_tri(A, B, N, M%(2*L), L/2);
        permute(A+(M%(2*L))*N, B+(M%(2*L))*N, N, M-M%(2*L), L);
    }
}

void permute_t_tri(float * A, const float * B, const int N, const int M, const int L) {
    if (L == 2) {
        permute_t(A, B, N, M, 2);
    }
    else {
        permute_t_tri(A, B, N, M%(2*L), L/2);
        permute_t(A+(M%(2*L))*N, B+(M%(2*L))*N, N, M-M%(2*L), L);
    }
}


void swap(float * A, float * B, const int N) {
    float tmp;
    for (int i = 0; i < N; i++) {
        tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
}

void warp(float * A, const int N, const int M, const int L) {
    for (int j = 2; j <= L; j <<= 1) {
        #pragma omp parallel for
        for (int i = M%(4*L); i < M; i += 4*j)
            swap(A+(i+j)*N, A+(i+j*2)*N, j*N);
    }
}

void warp_t(float * A, const int N, const int M, const int L) {
    for (int j = L; j >= 2; j >>= 1) {
        #pragma omp parallel for
        for (int i = M%(4*L); i < M; i += 4*j)
            swap(A+(i+j)*N, A+(i+j*2)*N, j*N);
    }
}
