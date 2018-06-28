// Permutations that enable SSE and AVX vectorization.

#include "fasttransforms.h"

void permute_sph_SSE(const double * A, double * B, const int N, const int M) {
    for (int i = 0; i < N; i++)
        B[i] = A[i];
    for (int j = 1; j < M; j += 2)
        for (int i = 0; i < 2*N; i++)
            B[(2*i)%(2*N)+(2*i)/(2*N)+j*N] = A[i+j*N];
}

void permute_t_sph_SSE(double * A, const double * B, const int N, const int M) {
    for (int i = 0; i < N; i++)
        A[i] = B[i];
    for (int j = 1; j < M; j += 2)
        for (int i = 0; i < 2*N; i++)
            A[i+j*N] = B[(2*i)%(2*N)+(2*i)/(2*N)+j*N];
}

void permute_sph_AVX(const double * A, double * B, const int N, const int M) {
    permute_sph_SSE(A, B, N, M%8);
    #pragma omp parallel for
    for (int j = M%8; j < M; j += 4)
        for (int i = 0; i < 4*N; i++)
            B[(4*i)%(4*N)+(4*i)/(4*N)+j*N] = A[i+j*N];
}

void permute_t_sph_AVX(double * A, const double * B, const int N, const int M) {
    permute_t_sph_SSE(A, B, N, M%8);
    #pragma omp parallel for
    for (int j = M%8; j < M; j += 4)
        for (int i = 0; i < 4*N; i++)
            A[i+j*N] = B[(4*i)%(4*N)+(4*i)/(4*N)+j*N];
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
    for (int i = 1; i < M; i += 16) {
         swap_AVX(A+(i+2)*M, A+(i+4)*M, M);
        swap_AVX(A+(i+4)*M, A+(i+8)*M, M);
        swap_AVX(A+(i+6)*M, A+(i+12)*M, M);
        swap_AVX(A+(i+10)*M, A+(i+12)*M, M);
    }
}
    
