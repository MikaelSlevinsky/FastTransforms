// Permutations that enable SSE and AVX vectorization.

#include "fasttransforms.h"

void permute(const double * A, double * B, const int N, const int M, const int L) {
    #pragma omp parallel for
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            B[(L*i)%(L*N)+(L*i)/(L*N)+j*N] = A[i+j*N];
}

void permute_t(double * A, const double * B, const int N, const int M, const int L) {
    #pragma omp parallel for
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            A[i+j*N] = B[(L*i)%(L*N)+(L*i)/(L*N)+j*N];
}


void permute_sph(const double * A, double * B, const int N, const int M, const int L) {
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

void permute_t_sph(double * A, const double * B, const int N, const int M, const int L) {
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

void permute_tri(const double * A, double * B, const int N, const int M, const int L) {
    if (L == 2) {
        permute(A, B, N, M, 2);
    }
    else {
        permute_tri(A, B, N, M%(2*L), L/2);
        permute(A+(M%(2*L))*N, B+(M%(2*L))*N, N, M-M%(2*L), L);
    }
}

void permute_t_tri(double * A, const double * B, const int N, const int M, const int L) {
    if (L == 2) {
        permute_t(A, B, N, M, 2);
    }
    else {
        permute_t_tri(A, B, N, M%(2*L), L/2);
        permute_t(A+(M%(2*L))*N, B+(M%(2*L))*N, N, M-M%(2*L), L);
    }
}

void permute_disk(const double * A, double * B, const int N, const int M, const int L) {permute_sph(A, B, N, M, L);}
void permute_t_disk(double * A, const double * B, const int N, const int M, const int L) {permute_t_sph(A, B, N, M, L);}

void permute_spinsph(const double * A, double * B, const int N, const int M, const int L) {permute_sph(A, B, N, M, L);}
void permute_t_spinsph(double * A, const double * B, const int N, const int M, const int L) {permute_t_sph(A, B, N, M, L);}



void swap(double * A, double * B, const int N) {
    double tmp;
    for (int i = 0; i < 2*N; i++) {
        tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
}

void swap_SSE(double * A, double * B, const int N) {
    __m128d tmp;
    for (int i = 0; i < N; i++) {
        tmp = vload2(A+2*i);
        vstore2(A+2*i, vload2(B+2*i));
        vstore2(B+2*i, tmp);
    }
}

// Note : Call gcc with -mavx flag for this feature
void swap_AVX(double * A, double * B, const int N) {
    __m256d tmp;
    double tmpd;
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


void two_warp(double * A, const int N, const int M) {
    #pragma omp parallel for
    for (int i = M%8; i < M; i += 8)
        swap_AVX(A+(i+2)*N, A+(i+4)*N, N);
}

void four_warp(double * A, const int N, const int M) {
    #pragma omp parallel for
    for (int i = M%16; i < M; i += 16) {
        swap_AVX(A+(i+2)*N, A+(i+4)*N, N);
        swap_AVX(A+(i+4)*N, A+(i+8)*N, N);
        swap_AVX(A+(i+6)*N, A+(i+12)*N, N);
        swap_AVX(A+(i+10)*N, A+(i+12)*N, N);
    }
}

void reverse_four_warp(double * A, const int N, const int M) {
    #pragma omp parallel for
    for (int i = M%16; i < M; i += 16) {
        swap_AVX(A+(i+12)*N, A+(i+10)*N, N);
        swap_AVX(A+(i+12)*N, A+(i+6)*N, N);
        swap_AVX(A+(i+8)*N, A+(i+4)*N, N);
        swap_AVX(A+(i+4)*N, A+(i+2)*N, N);
    }
}
