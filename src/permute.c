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

void permute_sph_AVX(const double * A, double * B, const int N, const int M) {
    for (int i = 0; i < N; i++)
        B[i] = A[i];
    for (int j = 1; j < M; j += 4)
        for (int i = 0; i < 4*N; i++)
            B[(4*i)%(4*N)+(4*i)/(4*N)+j*N] = A[i+j*N];
}

void permute_t_sph_AVX(double * A, const double * B, const int N, const int M) {
    for (int i = 0; i < N; i++)
        B[i] = A[i];
    for (int j = 1; j < M; j += 4)
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

void swap(double * A, double * B, const int M){
    double tmp = 0.0;
    for(int i = 0; i < 2*M; i++){

        tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
}

void swap_SSE(double * A, double * B, const int M){
    __m128d tmp1;
    __m128d tmp2;
    for(int i = 0; i < M; i++){
        
        tmp1 = vload2(A+2*i);
        tmp2 = vload2(B+2*i);
        vstore2((A+i*2), tmp2);
        vstore2((B+i*2), tmp1);
    }
}

// Note : Call gcc with -mavx flag for this feature
void swap_AVX(double * A, double * B, const int N){
    __m256d tmp1;
    __m256d tmp2;
    for(int i = 0; i < N/2; i++){
        
        tmp1 = vload4(A+i*4);
        tmp2 = vload4(B+i*4);
        vstore4((A+i*4), tmp2);
        vstore4((B+i*4), tmp1);
        
    }
    
    if(N%2 != 0){
        double tmpd = 0.0;
            
        tmpd = A[2*N-2];
        A[2*N-2] = B[2*N-2];
        B[2*N-2] = tmpd;
            
        tmpd = A[2*N-1];
        A[2*N-1] = B[2*N-1];
        B[2*N-1] = tmpd; 
    }
    
}

//swap_AVX(A+(i+2)*M, A+(i+4)*M, M);
//will give 0,1 ordering, instead of 0,2
void two_warp(const int N, const int M, double * A){
    for (int i = 1; i < M; i+=8){
        swap_AVX(A+(i+2)*N, A+(i+4)*N, N);     
    }
}

void four_warp(const int N, const int M, double * A){
    for (int i = 1; i < M; i+=16){
            
        swap_AVX(A+(i+2)*M, A+(i+4)*M, M);
        swap_AVX(A+(i+4)*M, A+(i+8)*M, M);
        swap_AVX(A+(i+6)*M, A+(i+12)*M, M);
        swap_AVX(A+(i+10)*M, A+(i+12)*M, M);
    }
}
