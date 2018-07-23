// Permutations that enable SSE and AVX vectorization.

#include "fasttransforms.h"

void permute(const double * A, double * B, const int N, const int M, const int L) {
    #pragma omp parallel for if (N < 2*M)
    for (int j = 0; j < M; j += L)
        for (int i = 0; i < L*N; i++)
            B[(L*i)%(L*N)+(L*i)/(L*N)+j*N] = A[i+j*N];
}

void permute_t(double * A, const double * B, const int N, const int M, const int L) {
    #pragma omp parallel for if (N < 2*M)
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

void warp(double * A, const int N, const int M, const int L){
    for (int j = 2;  j <= L; j *= 2) {
        #pragma omp parallel for
        for (int i = M%(4*L); i < M; i += (4*j))
            swap_AVX(A+(i+j)*N, A+(i+j*2)*N, N*(j/2));
    }
}

void reverse_warp(double * A, const int N, const int M, const int L){
    for (int j = L;  j >= 2; j = j/2) {
        #pragma omp parallel for
        for (int i = M%(4*L); i < M; i += (4*j))
            swap_AVX(A+(i+j)*N, A+(i+j*2)*N, N*(j/2));
    }
}
