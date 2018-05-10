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


#define VECTOR_SIZE_2 2
typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
#define vall2(x) ((double2) _mm_set1_pd(x))
#define vload2(v) ((double2) _mm_loadu_pd(v))
#define vstore2(u, v) (_mm_storeu_pd(u, v))
#define vread2(a, b) ((double2) _mm_set_pd(a, b))
#define vwrite2(a, b, v) { \
                         _mm_storeh_pd(&a, v); \
                         _mm_storel_pd(&b, v); \
                        }

#define VECTOR_SIZE_4 4
typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
#define vall4(x) ((double4) _mm256_set1_pd(x))
#define vload4(v) ((double4) _mm256_loadu_pd(v))
#define vstore4(u, v) (_mm256_storeu_pd(u, v))
#define vread4(a, b, c, d) ((double4) _mm256_set_pd(a, b, c, d))
#define vwrite4(a, b, c, d, v) { \
                               __m128d t1 = ((__m128d) _mm256_extractf128_pd(v, 1)); \
                               _mm_storeh_pd(&a, t1); \
                               _mm_storel_pd(&b, t1); \
                               __m128d t2 = ((__m128d) _mm256_extractf128_pd(v, 0)); \
                               _mm_storeh_pd(&c, t2); \
                               _mm_storel_pd(&d, t2); \
                              }


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
RotationPlan * plan_rotspinsphere(const int n, const int m1, const int m2);
RotationPlan * plan_rotdisk(const int n);
RotationPlan * plan_rottriangle(const int n, const double alpha, const double beta, const double gamma);

void kernel1_sph_hi2lo(const RotationPlan * RP, const int m, double * A);
void kernel1_sph_lo2hi(const RotationPlan * RP, const int m, double * A);

void kernel2_sph_hi2lo(const RotationPlan * RP, const int m, double * A, double * B);
void kernel2_sph_lo2hi(const RotationPlan * RP, const int m, double * A, double * B);

void kernel2x4_sph_hi2lo(const RotationPlan * RP, const int m, double * A, double * B);
void kernel2x4_sph_lo2hi(const RotationPlan * RP, const int m, double * A, double * B);

void kernel1_tri_hi2lo(const RotationPlan * RP, const int m, double * A);
void kernel1_tri_lo2hi(const RotationPlan * RP, const int m, double * A);

static inline void apply_givens_1x1(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A);
static inline void apply_givens_t_1x1(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A);

static inline void apply_givens_2x1(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);
static inline void apply_givens_t_2x1(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);

static inline void apply_givens_2x2(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);
static inline void apply_givens_t_2x2(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);


void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M);

void execute_tri_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_tri_lo2hi(const RotationPlan * RP, double * A, const int M);

typedef struct {
    RotationPlan * RP;
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

#endif //FASTTRANSFORMS_H
