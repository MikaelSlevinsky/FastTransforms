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
    for (int i = 0; i < 2*N; i++) {
        tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
}

/*
void swap_SSE(float * A, float * B, const int N) {
    __m128d tmp;
    for (int i = 0; i < N; i++) {
        tmp = vload2(A+2*i);
        vstore2(A+2*i, vload2(B+2*i));
        vstore2(B+2*i, tmp);
    }
}

// Note : Call gcc with -mavx flag for this feature
void swap_AVX(float * A, float * B, const int N) {
    __m256d tmp;
    float tmpd;
    for (int i = 0; i < N/2; i++) {
        tmp = vload4(A+4*i);
        vstore4(A+4*i, vload4(B+4*i));
        vstore4(B+4*i, tmp);
    }
    if (N%2 != 0) {
        tmpd = A[2*N-2];
        A[2*N-2] = B[2*N-2];
        B[2*N-2] = tmpd;
        tmpd = A[2*N-1];
        A[2*N-1] = B[2*N-1];
        B[2*N-1] = tmpd;
    }
}
*/


void two_warp(float * A, const int N, const int M) {
    #pragma omp parallel for
    for (int i = M%8; i < M; i += 8)
        swap(A+(i+2)*N, A+(i+4)*N, N);
}

void four_warp(float * A, const int N, const int M) {
    #pragma omp parallel for
    for (int i = M%16; i < M; i += 16) {
        swap(A+(i+2)*N, A+(i+4)*N, N);
        swap(A+(i+4)*N, A+(i+8)*N, N);
        swap(A+(i+6)*N, A+(i+12)*N, N);
        swap(A+(i+10)*N, A+(i+12)*N, N);
    }
}

void reverse_four_warp(float * A, const int N, const int M) {
    #pragma omp parallel for
    for (int i = M%16; i < M; i += 16) {
        swap(A+(i+12)*N, A+(i+10)*N, N);
        swap(A+(i+12)*N, A+(i+6)*N, N);
        swap(A+(i+8)*N, A+(i+4)*N, N);
        swap(A+(i+4)*N, A+(i+2)*N, N);
    }
}
