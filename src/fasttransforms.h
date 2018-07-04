// Computational routines for one-dimensional orthogonal polynomial transforms.

#ifndef FASTTRANSFORMS_H
#define FASTTRANSFORMS_H

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


#if __SSE2__
    #define VECTOR_SIZE_2 2
    typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
    #define vall2(x) ((double2) _mm_set1_pd(x))
    #define vload2(v) ((double2) _mm_loadu_pd(v))
    #define vstore2(u, v) (_mm_storeu_pd(u, v))
#endif
#if __AVX__
    #define VECTOR_SIZE_4 4
    typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
    #define vall4(x) ((double4) _mm256_set1_pd(x))
    #define vload4(v) ((double4) _mm256_loadu_pd(v))
    #define vstore4(u, v) (_mm256_storeu_pd(u, v))
#endif
#if __AVX512F__
    #define VECTOR_SIZE_8 8
    typedef double double8 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
    #define vall8(x) ((double8) _mm512_set1_pd(x))
    #define vload8(v) ((double8) _mm512_loadu_pd(v))
    #define vstore8(u, v) (_mm512_storeu_pd(u, v))
#endif

static inline double stirlingseries(const double z);
static inline double Aratio(const int n, const double alpha, const double beta);
static inline double Analphabeta(const int n, const double alpha, const double beta);
static inline double lambda(const double x);
static inline double lambda2(const double x, const double l1, const double l2);

double * plan_leg2cheb(const int normleg, const int normcheb, const int n);
double * plan_cheb2leg(const int normcheb, const int normleg, const int n);
double * plan_ultra2ultra(const int normultra1, const int normultra2, const int n, const double lambda1, const double lambda2);
double * plan_jac2jac(const int normjac1, const int normjac2, const int n, const double alpha, const double beta, const double gamma);


typedef struct {
    double * s;
    double * c;
    int n;
} RotationPlan;

RotationPlan * plan_rotsphere(const int n);

void kernel_sph_hi2lo(const RotationPlan * RP, const int m, double * A);
void kernel_sph_lo2hi(const RotationPlan * RP, const int m, double * A);

void kernel_sph_hi2lo_SSE(const RotationPlan * RP, const int m, double * A);
void kernel_sph_lo2hi_SSE(const RotationPlan * RP, const int m, double * A);

void kernel_sph_hi2lo_AVX(const RotationPlan * RP, const int m, double * A);
void kernel_sph_lo2hi_AVX(const RotationPlan * RP, const int m, double * A);

void kernel_sph_hi2lo_AVX512(const RotationPlan * RP, const int m, double * A);
void kernel_sph_lo2hi_AVX512(const RotationPlan * RP, const int m, double * A);

RotationPlan * plan_rottriangle(const int n, const double alpha, const double beta, const double gamma);

void kernel_tri_hi2lo(const RotationPlan * RP, const int m, double * A);
void kernel_tri_lo2hi(const RotationPlan * RP, const int m, double * A);

void kernel_tri_hi2lo_SSE(const RotationPlan * RP, const int m, double * A);
void kernel_tri_lo2hi_SSE(const RotationPlan * RP, const int m, double * A);

void kernel_tri_hi2lo_AVX(const RotationPlan * RP, const int m, double * A);
void kernel_tri_lo2hi_AVX(const RotationPlan * RP, const int m, double * A);

void kernel_tri_hi2lo_AVX512(const RotationPlan * RP, const int m, double * A);
void kernel_tri_lo2hi_AVX512(const RotationPlan * RP, const int m, double * A);

RotationPlan * plan_rotdisk(const int n);

void kernel_disk_hi2lo(const RotationPlan * RP, const int m, double * A);
void kernel_disk_lo2hi(const RotationPlan * RP, const int m, double * A);

void kernel_disk_hi2lo_SSE(const RotationPlan * RP, const int m, double * A);
void kernel_disk_lo2hi_SSE(const RotationPlan * RP, const int m, double * A);

static inline void apply_givens(const double S, const double C, double * X, double * Y);
static inline void apply_givens_t(const double S, const double C, double * X, double * Y);

static inline void apply_givens_SSE(const double S, const double C, double * X, double * Y);
static inline void apply_givens_t_SSE(const double S, const double C, double * X, double * Y);

static inline void apply_givens_AVX(const double S, const double C, double * X, double * Y);
static inline void apply_givens_t_AVX(const double S, const double C, double * X, double * Y);

static inline void apply_givens_AVX512(const double S, const double C, double * X, double * Y);
static inline void apply_givens_t_AVX512(const double S, const double C, double * X, double * Y);

void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M);

void execute_sph_hi2lo_SSE(const RotationPlan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_SSE(const RotationPlan * RP, double * A, double * B, const int M);

void execute_sph_hi2lo_AVX(const RotationPlan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_AVX(const RotationPlan * RP, double * A, double * B, const int M);

void execute_sph_hi2lo_AVX512(const RotationPlan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_AVX512(const RotationPlan * RP, double * A, double * B, const int M);

void execute_tri_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_tri_lo2hi(const RotationPlan * RP, double * A, const int M);

void execute_tri_hi2lo_SSE(const RotationPlan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_SSE(const RotationPlan * RP, double * A, double * B, const int M);

void execute_tri_hi2lo_AVX(const RotationPlan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_AVX(const RotationPlan * RP, double * A, double * B, const int M);

void execute_tri_hi2lo_AVX512(const RotationPlan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_AVX512(const RotationPlan * RP, double * A, double * B, const int M);

void execute_disk_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_disk_lo2hi(const RotationPlan * RP, double * A, const int M);

void execute_disk_hi2lo_SSE(const RotationPlan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_SSE(const RotationPlan * RP, double * A, double * B, const int M);

typedef struct {
    RotationPlan * RP;
    double * B;
    double * P1;
    double * P2;
    double * P1inv;
    double * P2inv;
} SphericalHarmonicPlan;

SphericalHarmonicPlan * plan_sph2fourier(const int n);

void execute_sph2fourier(const SphericalHarmonicPlan * P, double * A, const int N, const int M);
void execute_fourier2sph(const SphericalHarmonicPlan * P, double * A, const int N, const int M);

typedef struct {
    RotationPlan * RP;
    double * P1;
    double * P2;
    double * P3;
    double * P4;
    double * P1inv;
    double * P2inv;
    double * P3inv;
    double * P4inv;
    double alpha;
    double beta;
    double gamma;
} TriangularHarmonicPlan;

TriangularHarmonicPlan * plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma);

void execute_tri2cheb(const TriangularHarmonicPlan * P, double * A, const int N, const int M);
void execute_cheb2tri(const TriangularHarmonicPlan * P, double * A, const int N, const int M);

static void alternate_sign(double * A, const int N);

static void chebyshev_normalization(double * A, const int N, const int M);
static void chebyshev_normalization_t(double * A, const int N, const int M);

void two_warp(double * A, const int N, const int M);
void four_warp(double * A, const int N, const int M);
void reverse_four_warp(double * A, const int N, const int M);

void swap(double * A, double * B, const int N);
void swap_SSE(double * A, double * B, const int N);
void swap_AVX(double * A, double * B, const int N);

void permute_sph_SSE(const double * A, double * B, const int N, const int M);
void permute_t_sph_SSE(double * A, const double * B, const int N, const int M);
void permute_sph_AVX(const double * A, double * B, const int N, const int M);
void permute_t_sph_AVX(double * A, const double * B, const int N, const int M);
void permute_sph_AVX512(const double * A, double * B, const int N, const int M);
void permute_t_sph_AVX512(double * A, const double * B, const int N, const int M);

void permute_tri_SSE(const double * A, double * B, const int N, const int M);
void permute_t_tri_SSE(double * A, const double * B, const int N, const int M);
void permute_tri_AVX(const double * A, double * B, const int N, const int M);
void permute_t_tri_AVX(double * A, const double * B, const int N, const int M);
void permute_tri_AVX512(const double * A, double * B, const int N, const int M);
void permute_t_tri_AVX512(double * A, const double * B, const int N, const int M);

void permute_disk_SSE(const double * A, double * B, const int N, const int M);
void permute_t_disk_SSE(double * A, const double * B, const int N, const int M);

#endif //FASTTRANSFORMS_H
