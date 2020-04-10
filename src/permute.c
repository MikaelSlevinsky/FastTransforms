// Permutations that enable SSE, AVX, and AVX-512 vectorization.

#include "ftinternal.h"

void permute(const double * A, double * B, const int N, const int M, const int L) {
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            B[(L*i)%(L*N)+(L*i)/(L*N)+j*N] = A[i+j*N];
}
void permute_t(double * A, const double * B, const int N, const int M, const int L) {
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            A[i+j*N] = B[(L*i)%(L*N)+(L*i)/(L*N)+j*N];
}

void permute_tri(const double * A, double * B, const int N, const int M, const int L) {
    for (int i = 0; i < (M%L)*N; i++)
        B[i] = A[i];
    permute(A+(M%L)*N, B+(M%L)*N, N, M-(M%L), L);
}

void permute_t_tri(double * A, const double * B, const int N, const int M, const int L) {
    for (int i = 0; i < (M%L)*N; i++)
        A[i] = B[i];
    permute_t(A+(M%L)*N, B+(M%L)*N, N, M-(M%L), L);
}

/*
void permute(const double * A, double * B, const int N, const int M, const int L) {
    int NB = VALIGN(N);
    #pragma omp parallel for if (N < 2*M)
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            B[(L*i)%(L*N)+(L*i)/(L*N)+j*NB] = A[i+j*N];
}

void permute_t(double * A, const double * B, const int N, const int M, const int L) {
    int NB = VALIGN(N);
    #pragma omp parallel for if (N < 2*M)
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            A[i+j*N] = B[(L*i)%(L*N)+(L*i)/(L*N)+j*NB];
}
*/

void permute_sph(const double * A, double * B, const int N, const int M, const int L) {
    int NB = VALIGN(N);
    if (L == 2) {
        for (int i = 0; i < N; i++)
            B[i] = A[i];
        permute(A+N, B+NB, N, M-1, 2);
    }
    else {
        permute_sph(A, B, N, M%(2*L), L/2);
        permute(A+(M%(2*L))*N, B+(M%(2*L))*NB, N, M-M%(2*L), L);
    }
}

void permute_t_sph(double * A, const double * B, const int N, const int M, const int L) {
    int NB = VALIGN(N);
    if (L == 2) {
        for (int i = 0; i < N; i++)
            A[i] = B[i];
        permute_t(A+N, B+NB, N, M-1, 2);
    }
    else {
        permute_t_sph(A, B, N, M%(2*L), L/2);
        permute_t(A+(M%(2*L))*N, B+(M%(2*L))*NB, N, M-M%(2*L), L);
    }
}

void old_permute_tri(const double * A, double * B, const int N, const int M, const int L) {
    int NB = VALIGN(N);
    if (L == 2) {
        if (M%2) {
            for (int i = 0; i < N; i++)
                B[i] = A[i];
            permute(A+N, B+NB, N, M-1, 2);
        } else {
            permute(A, B, N, M, 2);
        }
    }
    else {
        old_permute_tri(A, B, N, M%(2*L), L/2);
        permute(A+(M%(2*L))*N, B+(M%(2*L))*NB, N, M-M%(2*L), L);
    }
}

void old_permute_t_tri(double * A, const double * B, const int N, const int M, const int L) {
    int NB = VALIGN(N);
    if (L == 2) {
        if (M%2) {
            for (int i = 0; i < N; i++)
                A[i] = B[i];
            permute_t(A+N, B+NB, N, M-1, 2);
        } else {
            permute_t(A, B, N, M, 2);
        }
    }
    else {
        old_permute_t_tri(A, B, N, M%(2*L), L/2);
        permute_t(A+(M%(2*L))*N, B+(M%(2*L))*NB, N, M-M%(2*L), L);
    }
}


void swap_warp(double * A, double * B, const int N) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return swap_warp_AVX512F(A, B, N);
    else if (simd.avx)
        return swap_warp_AVX(A, B, N);
    else if (simd.sse2)
        return swap_warp_SSE2(A, B, N);
    else
        return swap_warp_default(A, B, N);
}

void swap_warpf(float * A, float * B, const int N) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return swap_warp_AVX512Ff(A, B, N);
    else if (simd.avx)
        return swap_warp_AVXf(A, B, N);
    else if (simd.sse)
        return swap_warp_SSEf(A, B, N);
    else
        return swap_warp_defaultf(A, B, N);
}

void warp(double * A, const int N, const int M, const int L) {
    for (int j = 2; j <= L; j <<= 1)
        for (int i = M%(4*L); i < M; i += 4*j)
            swap_warp(A+(i+j)*N, A+(i+2*j)*N, j*N);
}

void warp_t(double * A, const int N, const int M, const int L) {
    for (int j = L; j >= 2; j >>= 1)
        for (int i = M%(4*L); i < M; i += 4*j)
            swap_warp(A+(i+j)*N, A+(i+2*j)*N, j*N);
}
