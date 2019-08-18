#ifndef FTINTERNAL_H
#define FTINTERNAL_H

#include <stdlib.h>
#include <math.h>
#include <quadmath.h>
#include <immintrin.h>

#define RED(string) "\x1b[31m" string "\x1b[0m"
#define GREEN(string) "\x1b[32m" string "\x1b[0m"
#define YELLOW(string) "\x1b[33m" string "\x1b[0m"
#define BLUE(string) "\x1b[34m" string "\x1b[0m"
#define MAGENTA(string) "\x1b[35m" string "\x1b[0m"
#define CYAN(string) "\x1b[36m" string "\x1b[0m"

#define M_SQRT_PI      1.772453850905516027   /* sqrt(pi)           */
#define M_1_SQRT_PI    0.564189583547756287   /* 1/sqrt(pi)         */
#define M_SQRT_PI_2    0.886226925452758014   /* sqrt(pi)/2         */
#define M_4_SQRT_PI    7.089815403622064109   /* 4*sqrt(pi)         */
#define M_1_4_SQRT_PI  0.141047395886939072   /* 1/(4*sqrt(pi))     */
#define M_PI_2_POW_0P5 1.253314137315500251   /* sqrt(pi/2)         */
#define M_2_PI_POW_0P5 0.797884560802865355   /* sqrt(2/pi)         */
#define M_PI_2_POW_1P5 1.968701243215302468   /* pow(pi/2, 1.5)     */
#define M_2_PI_POW_1P5 0.507949087473927758   /* pow(2/pi, 1.5)     */
#define M_EPSf         0x1p-23f               /* powf(2.0f, -23)    */
#define M_EPS          0x1p-52                /* pow(2.0, -52)      */
#define M_EPSl         0x1p-64l               /* powl(2.0l, -64)    */
#define M_EPSq         0x1p-112q              /* powq(2.0q, -112)   */
#define M_FLT_MINf     0x1p-126f              /* powf(2.0f, -126)   */
#define M_FLT_MIN      0x1p-1022              /* pow(2.0, -1022)    */
#define M_FLT_MINl     0x1p-16382l            /* powl(2.0l, -16382) */
#define M_FLT_MINq     0x1p-16382q            /* powq(2.0q, -16382) */

#define M_PIf          0xc.90fdaap-2f         // 3.1415927f0
#ifndef M_PIl
    #define M_PIl      0xc.90fdaa22168c235p-2l
#endif

#ifndef M_PI_2l
    #define M_PI_2l     0xc.90fdaa22168c235p-3L
#endif

#ifndef M_1_PIl
    #define M_1_PIl     0xa.2f9836e4e44152ap-5L
#endif

#ifndef M_2_PIl
    #define M_2_PIl     0xa.2f9836e4e44152ap-4L
#endif

typedef __float128 quadruple;

static inline float epsf(void) {return M_EPSf;}
static inline double eps(void) {return M_EPS;}
static inline long double epsl(void) {return M_EPSl;}
static inline quadruple epsq(void) {return M_EPSq;}

static inline float floatminf(void) {return M_FLT_MINf;}
static inline double floatmin(void) {return M_FLT_MIN;}
static inline long double floatminl(void) {return M_FLT_MINl;}
static inline quadruple floatminq(void) {return M_FLT_MINq;}

#if !(__APPLE__)
    static inline float __cospif(float x) {return cosf(M_PIf*x);}
    static inline double __cospi(double x) {return cos(M_PI*x);}
    static inline float __sinpif(float x) {return sinf(M_PIf*x);}
    static inline double __sinpi(double x) {return sin(M_PI*x);}
    static inline float __tanpif(float x) {return tanf(M_PIf*x);}
    static inline double __tanpi(double x) {return tan(M_PI*x);}
#endif
static inline long double __cospil(long double x) {return cosl(M_PIl*x);}
static inline quadruple __cospiq(quadruple x) {return cosq(M_PIq*x);}
static inline long double __sinpil(long double x) {return sinl(M_PIl*x);}
static inline quadruple __sinpiq(quadruple x) {return sinq(M_PIq*x);}
static inline long double __tanpil(long double x) {return tanl(M_PIl*x);}
static inline quadruple __tanpiq(quadruple x) {return tanq(M_PIq*x);}

#define ZERO(FLT) ((FLT) 0)
#define ONE(FLT) ((FLT) 1)
#define TWO(FLT) ((FLT) 2)

#define isfinitef(x) isfinite(x)
#define isfinitel(x) isfinite(x)
#define isfiniteq(x) finiteq(x)

#define isinff(x) isinf(x)
#define isinfl(x) isinf(x)

#define isnanf(x) isnan(x)
#define isnanl(x) isnan(x)

#define signbitf(x) signbit(x)
#define signbitl(x) signbit(x)

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

#define VALIGN(N) ((N + ALIGN_SIZE - 1) & -ALIGN_SIZE)
#define VMALLOC(s) _mm_malloc(s, ALIGN_SIZE*8)
#define VFREE(s) _mm_free(s)

double * plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n);
double * plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n);
double * plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, const double lambda, const double mu);
double * plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const double alpha, const double beta, const double gamma, const double delta);
double * plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const double alpha, const double beta);
double * plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, const double alpha, const double beta, const double lambda);
double * plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, const double lambda, const double alpha, const double beta);
double * plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, const double alpha, const double beta);
double * plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, const double alpha, const double beta);
double * plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, const double lambda);
double * plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, const double lambda);

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

void swap_warp(double * A, double * B, const int N);
void warp(double * A, const int N, const int M, const int L);
void warp_t(double * A, const int N, const int M, const int L);

// A bitwise OR ('|') of zero or more of the following: FFTW_ESTIMATE FFTW_MEASURE FFTW_PATIENT FFTW_EXHAUSTIVE FFTW_WISDOM_ONLY FFTW_DESTROY_INPUT FFTW_PRESERVE_INPUT FFTW_UNALIGNED
#define FT_FFTW_FLAGS FFTW_MEASURE | FFTW_DESTROY_INPUT

#endif //FTINTERNAL_H
