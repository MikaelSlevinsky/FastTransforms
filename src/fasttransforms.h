// Computational routines for one-dimensional orthogonal polynomial transforms.

#ifndef FASTTRANSFORMS_H
#define FASTTRANSFORMS_H

#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <cblas.h>
#include <fftw3.h>

#ifdef _OPENMP
    #include <omp.h>
    #define FT_GET_THREAD_NUM() omp_get_thread_num()
    #define FT_GET_NUM_THREADS() omp_get_num_threads()
    #define FT_GET_MAX_THREADS() omp_get_max_threads()
    #define FT_SET_NUM_THREADS(x) omp_set_num_threads(x)
#else
    #define FT_GET_THREAD_NUM() 0
    #define FT_GET_NUM_THREADS() 1
    #define FT_GET_MAX_THREADS() 1
    #define FT_SET_NUM_THREADS(x)
#endif

#define M_SQRT_PI      1.772453850905516027   /* sqrt(pi)       */
#define M_1_SQRT_PI    0.564189583547756287   /* 1/sqrt(pi)     */
#define M_SQRT_PI_2    0.886226925452758014   /* sqrt(pi)/2     */
#define M_4_SQRT_PI    7.089815403622064109   /* 4*sqrt(pi)     */
#define M_1_4_SQRT_PI  0.141047395886939072   /* 1/(4*sqrt(pi)) */
#define M_EPS          2.220446049250313E-16  /* pow(2.0, -52)  */

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#if __AVX512F__
    #define VECTOR_SIZE_8 8
    #define ALIGN_SIZE VECTOR_SIZE_8
    typedef double double8 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
    #define vall8(x) ((double8) _mm512_set1_pd(x))
    #define vload8(v) ((double8) _mm512_load_pd(v))
    #define vstore8(u, v) (_mm512_store_pd(u, v))
#endif
#if __AVX__
    #define VECTOR_SIZE_4 4
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_4
    #endif
    typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
    #define vall4(x) ((double4) _mm256_set1_pd(x))
    #define vload4(v) ((double4) _mm256_load_pd(v))
    #define vstore4(u, v) (_mm256_store_pd(u, v))
#endif
#if __SSE2__
    #define VECTOR_SIZE_2 2
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_2
    #endif
    typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
    #define vall2(x) ((double2) _mm_set1_pd(x))
    #define vload2(v) ((double2) _mm_load_pd(v))
    #define vstore2(u, v) (_mm_store_pd(u, v))
#endif

#define ALIGNB(N) (N + ALIGN_SIZE-1-(N+ALIGN_SIZE-1)%ALIGN_SIZE)
#define VMALLOC(s) _mm_malloc(s, ALIGN_SIZE*8)
#define VFREE(s) _mm_free(s)

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

void freeRotationPlan(RotationPlan * RP);

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

void kernel_disk_hi2lo_AVX(const RotationPlan * RP, const int m, double * A);
void kernel_disk_lo2hi_AVX(const RotationPlan * RP, const int m, double * A);

void kernel_disk_hi2lo_AVX512(const RotationPlan * RP, const int m, double * A);
void kernel_disk_lo2hi_AVX512(const RotationPlan * RP, const int m, double * A);


typedef struct {
    double * s1;
    double * c1;
    double * s2;
    double * c2;
    double * s3;
    double * c3;
    int n;
    int s;
} SpinRotationPlan;

void freeSpinRotationPlan(SpinRotationPlan * SRP);

SpinRotationPlan * plan_rotspinsphere(const int n, const int s);

void kernel_spinsph_hi2lo(const SpinRotationPlan * SRP, const int m, double * A);
void kernel_spinsph_lo2hi(const SpinRotationPlan * SRP, const int m, double * A);

void kernel_spinsph_hi2lo_SSE(const SpinRotationPlan * SRP, const int m, double * A);
void kernel_spinsph_lo2hi_SSE(const SpinRotationPlan * SRP, const int m, double * A);

void kernel_spinsph_hi2lo_AVX(const SpinRotationPlan * SRP, const int m, double * A);
void kernel_spinsph_lo2hi_AVX(const SpinRotationPlan * SRP, const int m, double * A);

void kernel_spinsph_hi2lo_AVX512(const SpinRotationPlan * SRP, const int m, double * A);
void kernel_spinsph_lo2hi_AVX512(const SpinRotationPlan * SRP, const int m, double * A);

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

void execute_disk_hi2lo_AVX(const RotationPlan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_AVX(const RotationPlan * RP, double * A, double * B, const int M);

void execute_disk_hi2lo_AVX512(const RotationPlan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_AVX512(const RotationPlan * RP, double * A, double * B, const int M);

void execute_spinsph_hi2lo(const SpinRotationPlan * SRP, double * A, const int M);
void execute_spinsph_lo2hi(const SpinRotationPlan * SRP, double * A, const int M);

void execute_spinsph_hi2lo_SSE(const SpinRotationPlan * SRP, double * A, double * B, const int M);
void execute_spinsph_lo2hi_SSE(const SpinRotationPlan * SRP, double * A, double * B, const int M);

void execute_spinsph_hi2lo_AVX(const SpinRotationPlan * SRP, double * A, double * B, const int M);
void execute_spinsph_lo2hi_AVX(const SpinRotationPlan * SRP, double * A, double * B, const int M);

void execute_spinsph_hi2lo_AVX512(const SpinRotationPlan * SRP, double * A, double * B, const int M);
void execute_spinsph_lo2hi_AVX512(const SpinRotationPlan * SRP, double * A, double * B, const int M);

typedef struct {
    RotationPlan * RP;
    double * B;
    double * P1;
    double * P2;
    double * P1inv;
    double * P2inv;
} SphericalHarmonicPlan;

void freeSphericalHarmonicPlan(SphericalHarmonicPlan * P);

SphericalHarmonicPlan * plan_sph2fourier(const int n);

void execute_sph2fourier(const SphericalHarmonicPlan * P, double * A, const int N, const int M);
void execute_fourier2sph(const SphericalHarmonicPlan * P, double * A, const int N, const int M);

typedef struct {
    RotationPlan * RP;
    double * B;
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

void freeTriangularHarmonicPlan(TriangularHarmonicPlan * P);

TriangularHarmonicPlan * plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma);

void execute_tri2cheb(const TriangularHarmonicPlan * P, double * A, const int N, const int M);
void execute_cheb2tri(const TriangularHarmonicPlan * P, double * A, const int N, const int M);

static void alternate_sign(double * A, const int N, const int M);
static void alternate_sign_t(double * A, const int N, const int M);

static void chebyshev_normalization(double * A, const int N, const int M);
static void chebyshev_normalization_t(double * A, const int N, const int M);


void permute(const double * A, double * B, const int N, const int M, const int L);
void permute_t(double * A, const double * B, const int N, const int M, const int L);

void permute_sph(const double * A, double * B, const int N, const int M, const int L);
void permute_t_sph(double * A, const double * B, const int N, const int M, const int L);

void permute_tri(const double * A, double * B, const int N, const int M, const int L);
void permute_t_tri(double * A, const double * B, const int N, const int M, const int L);

#define permute_disk(A, B, N, M, L) permute_sph(A, B, N, M, L)
#define permute_t_disk(A, B, N, M, L) permute_t_sph(A, B, N, M, L)

#define permute_spinsph(A, B, N, M, L) permute_sph(A, B, N, M, L)
#define permute_t_spinsph(A, B, N, M, L) permute_t_sph(A, B, N, M, L)

void swap(double * A, double * B, const int N);
void warp(double * A, const int N, const int M, const int L);
void warp_t(double * A, const int N, const int M, const int L);


typedef struct {
    fftw_plan plantheta1;
    fftw_plan plantheta2;
    fftw_plan plantheta3;
    fftw_plan plantheta4;
    fftw_plan planphi;
    double * Y;
} SphereFFTWPlan;

void freeSphereFFTWPlan(SphereFFTWPlan * P);

SphereFFTWPlan * plan_sph_synthesis(const int N, const int M);
SphereFFTWPlan * plan_sph_analysis(const int N, const int M);

void execute_sph_synthesis(const SphereFFTWPlan * P, double * X, const int N, const int M);
void execute_sph_analysis(const SphereFFTWPlan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planxy;
} TriangleFFTWPlan;

void freeTriangleFFTWPlan(TriangleFFTWPlan * P);

TriangleFFTWPlan * plan_tri_synthesis(const int N, const int M);
TriangleFFTWPlan * plan_tri_analysis(const int N, const int M);

void execute_tri_synthesis(const TriangleFFTWPlan * P, double * X, const int N, const int M);
void execute_tri_analysis(const TriangleFFTWPlan * P, double * X, const int N, const int M);


static inline void colswap(const double * X, double * Y, const int N, const int M);
static inline void colswap_t(double * X, const double * Y, const int N, const int M);

#endif //FASTTRANSFORMS_H
