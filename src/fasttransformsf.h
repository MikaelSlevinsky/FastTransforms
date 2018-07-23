#ifndef FASTTRANSFORMSF_H
#define FASTTRANSFORMSF_H

#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <cblas.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_num_threads() 1
#endif

#define M_SQRT_PI    1.772453850905516027   /* sqrt(pi)       */
#define M_1_SQRT_PI  0.564189583547756287   /* 1/sqrt(pi)     */
#define M_SQRT_PI_2  0.886226925452758014   /* sqrt(pi)/2     */

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

#if __SSE2__
    typedef float float4 __attribute__ ((vector_size (4*4)));
    #define vsetf4(x) ((float4) _mm_set1_ps(x))
    #define vloadf4(v) ((float4) _mm_loadu_ps(v))
    #define vstoref4(u, v) (_mm_storeu_ps(u, v))
#endif
#if __AVX__
    typedef float float8 __attribute__ ((vector_size (8*4)));
    #define vsetf8(x) ((float8) _mm256_set1_ps(x))
    #define vloadf8(v) ((float8) _mm256_loadu_ps(v))
    #define vstoref8(u, v) (_mm256_storeu_ps(u, v))
#endif
#if __AVX512F__
    typedef float float16 __attribute__ ((vector_size (16*4)));
    #define vsetf16(x) ((float16) _mm512_set1_ps(x))
    #define vloadf16(v) ((float16) _mm512_loadu_ps(v))
    #define vstoref16(u, v) (_mm512_storeu_ps(u, v))
#endif

static inline float stirlingseries(const float z);
static inline float Aratio(const int n, const float alpha, const float beta);
static inline float Analphabeta(const int n, const float alpha, const float beta);
static inline float lambda(const float x);
static inline float lambda2(const float x, const float l1, const float l2);

float * plan_leg2cheb(const int normleg, const int normcheb, const int n);
float * plan_cheb2leg(const int normcheb, const int normleg, const int n);
float * plan_ultra2ultra(const int normultra1, const int normultra2, const int n, const float lambda1, const float lambda2);
float * plan_jac2jac(const int normjac1, const int normjac2, const int n, const float alpha, const float beta, const float gamma);


typedef struct {
    float * s;
    float * c;
    int n;
} RotationPlan;

void freeRotationPlan(RotationPlan * RP);

RotationPlan * plan_rotsphere(const int n);

void kernel_sph_hi2lo(const RotationPlan * RP, const int m, float * A);
void kernel_sph_lo2hi(const RotationPlan * RP, const int m, float * A);

void kernel_sph_hi2lo_SSE(const RotationPlan * RP, const int m, float * A);
void kernel_sph_lo2hi_SSE(const RotationPlan * RP, const int m, float * A);

void kernel_sph_hi2lo_AVX(const RotationPlan * RP, const int m, float * A);
void kernel_sph_lo2hi_AVX(const RotationPlan * RP, const int m, float * A);

void kernel_sph_hi2lo_AVX512(const RotationPlan * RP, const int m, float * A);
void kernel_sph_lo2hi_AVX512(const RotationPlan * RP, const int m, float * A);

RotationPlan * plan_rottriangle(const int n, const float alpha, const float beta, const float gamma);

void kernel_tri_hi2lo(const RotationPlan * RP, const int m, float * A);
void kernel_tri_lo2hi(const RotationPlan * RP, const int m, float * A);

void kernel_tri_hi2lo_SSE(const RotationPlan * RP, const int m, float * A);
void kernel_tri_lo2hi_SSE(const RotationPlan * RP, const int m, float * A);

void kernel_tri_hi2lo_AVX(const RotationPlan * RP, const int m, float * A);
void kernel_tri_lo2hi_AVX(const RotationPlan * RP, const int m, float * A);

void kernel_tri_hi2lo_AVX512(const RotationPlan * RP, const int m, float * A);
void kernel_tri_lo2hi_AVX512(const RotationPlan * RP, const int m, float * A);

RotationPlan * plan_rotdisk(const int n);

void kernel_disk_hi2lo(const RotationPlan * RP, const int m, float * A);
void kernel_disk_lo2hi(const RotationPlan * RP, const int m, float * A);

void kernel_disk_hi2lo_SSE(const RotationPlan * RP, const int m, float * A);
void kernel_disk_lo2hi_SSE(const RotationPlan * RP, const int m, float * A);

void kernel_disk_hi2lo_AVX(const RotationPlan * RP, const int m, float * A);
void kernel_disk_lo2hi_AVX(const RotationPlan * RP, const int m, float * A);

void kernel_disk_hi2lo_AVX512(const RotationPlan * RP, const int m, float * A);
void kernel_disk_lo2hi_AVX512(const RotationPlan * RP, const int m, float * A);


typedef struct {
    float * s1;
    float * c1;
    float * s2;
    float * c2;
    float * s3;
    float * c3;
    int n;
    int s;
} SpinRotationPlan;

void freeSpinRotationPlan(SpinRotationPlan * SRP);

SpinRotationPlan * plan_rotspinsphere(const int n, const int s);

void kernel_spinsph_hi2lo(const SpinRotationPlan * SRP, const int m, float * A);
void kernel_spinsph_lo2hi(const SpinRotationPlan * SRP, const int m, float * A);

void kernel_spinsph_hi2lo_SSE(const SpinRotationPlan * SRP, const int m, float * A);
void kernel_spinsph_lo2hi_SSE(const SpinRotationPlan * SRP, const int m, float * A);

void kernel_spinsph_hi2lo_AVX(const SpinRotationPlan * SRP, const int m, float * A);
void kernel_spinsph_lo2hi_AVX(const SpinRotationPlan * SRP, const int m, float * A);

static inline void apply_givens(const float S, const float C, float * X, float * Y);
static inline void apply_givens_t(const float S, const float C, float * X, float * Y);

static inline void apply_givens_SSE(const float S, const float C, float * X, float * Y);
static inline void apply_givens_t_SSE(const float S, const float C, float * X, float * Y);

static inline void apply_givens_AVX(const float S, const float C, float * X, float * Y);
static inline void apply_givens_t_AVX(const float S, const float C, float * X, float * Y);

static inline void apply_givens_AVX512(const float S, const float C, float * X, float * Y);
static inline void apply_givens_t_AVX512(const float S, const float C, float * X, float * Y);

void execute_sph_hi2lo(const RotationPlan * RP, float * A, const int M);
void execute_sph_lo2hi(const RotationPlan * RP, float * A, const int M);

void execute_sph_hi2lo_SSE(const RotationPlan * RP, float * A, float * B, const int M);
void execute_sph_lo2hi_SSE(const RotationPlan * RP, float * A, float * B, const int M);

void execute_sph_hi2lo_AVX(const RotationPlan * RP, float * A, float * B, const int M);
void execute_sph_lo2hi_AVX(const RotationPlan * RP, float * A, float * B, const int M);

void execute_sph_hi2lo_AVX512(const RotationPlan * RP, float * A, float * B, const int M);
void execute_sph_lo2hi_AVX512(const RotationPlan * RP, float * A, float * B, const int M);

void execute_tri_hi2lo(const RotationPlan * RP, float * A, const int M);
void execute_tri_lo2hi(const RotationPlan * RP, float * A, const int M);

void execute_tri_hi2lo_SSE(const RotationPlan * RP, float * A, float * B, const int M);
void execute_tri_lo2hi_SSE(const RotationPlan * RP, float * A, float * B, const int M);

void execute_tri_hi2lo_AVX(const RotationPlan * RP, float * A, float * B, const int M);
void execute_tri_lo2hi_AVX(const RotationPlan * RP, float * A, float * B, const int M);

void execute_tri_hi2lo_AVX512(const RotationPlan * RP, float * A, float * B, const int M);
void execute_tri_lo2hi_AVX512(const RotationPlan * RP, float * A, float * B, const int M);

void execute_disk_hi2lo(const RotationPlan * RP, float * A, const int M);
void execute_disk_lo2hi(const RotationPlan * RP, float * A, const int M);

void execute_disk_hi2lo_SSE(const RotationPlan * RP, float * A, float * B, const int M);
void execute_disk_lo2hi_SSE(const RotationPlan * RP, float * A, float * B, const int M);

void execute_disk_hi2lo_AVX(const RotationPlan * RP, float * A, float * B, const int M);
void execute_disk_lo2hi_AVX(const RotationPlan * RP, float * A, float * B, const int M);

void execute_disk_hi2lo_AVX512(const RotationPlan * RP, float * A, float * B, const int M);
void execute_disk_lo2hi_AVX512(const RotationPlan * RP, float * A, float * B, const int M);

void execute_spinsph_hi2lo(const SpinRotationPlan * SRP, float * A, const int M);
void execute_spinsph_lo2hi(const SpinRotationPlan * SRP, float * A, const int M);

void execute_spinsph_hi2lo_SSE(const SpinRotationPlan * SRP, float * A, float * B, const int M);
void execute_spinsph_lo2hi_SSE(const SpinRotationPlan * SRP, float * A, float * B, const int M);

void execute_spinsph_hi2lo_AVX(const SpinRotationPlan * SRP, float * A, float * B, const int M);
void execute_spinsph_lo2hi_AVX(const SpinRotationPlan * SRP, float * A, float * B, const int M);

typedef struct {
    RotationPlan * RP;
    float * B;
    float * P1;
    float * P2;
    float * P1inv;
    float * P2inv;
} SphericalHarmonicPlan;

void freeSphericalHarmonicPlan(SphericalHarmonicPlan * P);

SphericalHarmonicPlan * plan_sph2fourier(const int n);

void execute_sph2fourier(const SphericalHarmonicPlan * P, float * A, const int N, const int M);
void execute_fourier2sph(const SphericalHarmonicPlan * P, float * A, const int N, const int M);

typedef struct {
    RotationPlan * RP;
    float * B;
    float * P1;
    float * P2;
    float * P3;
    float * P4;
    float * P1inv;
    float * P2inv;
    float * P3inv;
    float * P4inv;
    float alpha;
    float beta;
    float gamma;
} TriangularHarmonicPlan;

void freeTriangularHarmonicPlan(TriangularHarmonicPlan * P);

TriangularHarmonicPlan * plan_tri2cheb(const int n, const float alpha, const float beta, const float gamma);

void execute_tri2cheb(const TriangularHarmonicPlan * P, float * A, const int N, const int M);
void execute_cheb2tri(const TriangularHarmonicPlan * P, float * A, const int N, const int M);

static void alternate_sign(float * A, const int N);

static void chebyshev_normalization(float * A, const int N, const int M);
static void chebyshev_normalization_t(float * A, const int N, const int M);


void permute(const float * A, float * B, const int N, const int M, const int L);
void permute_t(float * A, const float * B, const int N, const int M, const int L);

void permute_sph(const float * A, float * B, const int N, const int M, const int L);
void permute_t_sph(float * A, const float * B, const int N, const int M, const int L);

void permute_tri(const float * A, float * B, const int N, const int M, const int L);
void permute_t_tri(float * A, const float * B, const int N, const int M, const int L);

#define permute_disk(A, B, N, M, L) permute_sph(A, B, N, M, L)
#define permute_t_disk(A, B, N, M, L) permute_t_sph(A, B, N, M, L)

#define permute_spinsph(A, B, N, M, L) permute_sph(A, B, N, M, L)
#define permute_t_spinsph(A, B, N, M, L) permute_t_sph(A, B, N, M, L)

void swap(float * A, float * B, const int N);
void warp(float * A, const int N, const int M, const int L);
void warp_t(float * A, const int N, const int M, const int L);

#endif //FASTTRANSFORMSF_H
