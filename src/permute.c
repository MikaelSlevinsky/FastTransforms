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

void permute_sph_AVX512(const double * A, double * B, const int N, const int M) {
    permute_sph_AVX(A, B, N, M%16);
    #pragma omp parallel for
    for (int j = M%16; j < M; j += 8)
        for (int i = 0; i < 8*N; i++)
            B[(8*i)%(8*N)+(8*i)/(8*N)+j*N] = A[i+j*N];
}

void permute_t_sph_AVX512(double * A, const double * B, const int N, const int M) {
    permute_t_sph_AVX(A, B, N, M%16);
    #pragma omp parallel for
    for (int j = M%16; j < M; j += 8)
        for (int i = 0; i < 8*N; i++)
            A[i+j*N] = B[(8*i)%(8*N)+(8*i)/(8*N)+j*N];
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

void permute_tri_AVX512(const double * A, double * B, const int N, const int M) {
    for (int j = 0; j < M; j += 8)
        for (int i = 0; i < 8*N; i++)
            B[(8*i)%(8*N)+(8*i)/(8*N)+j*N] = A[i+j*N];
}

void permute_t_tri_AVX512(double * A, const double * B, const int N, const int M) {
    for (int j = 0; j < M; j += 8)
        for (int i = 0; i < 8*N; i++)
            A[i+j*N] = B[(8*i)%(8*N)+(8*i)/(8*N)+j*N];
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
