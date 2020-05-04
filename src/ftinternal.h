#ifndef FTINTERNAL_H
#define FTINTERNAL_H

#include <stdlib.h>
#include <math.h>
#include <quadmath.h>
#include <immintrin.h>
#include <cpuid.h>

#define RED(string) "\x1b[31m" string "\x1b[0m"
#define GREEN(string) "\x1b[32m" string "\x1b[0m"
#define YELLOW(string) "\x1b[33m" string "\x1b[0m"
#define BLUE(string) "\x1b[34m" string "\x1b[0m"
#define MAGENTA(string) "\x1b[35m" string "\x1b[0m"
#define CYAN(string) "\x1b[36m" string "\x1b[0m"

#define M_SQRT_PI      1.772453850905516027   /* sqrt(pi)           */
#define M_1_SQRT_PI    0.564189583547756287   /* 1/sqrt(pi)         */
#define M_SQRT_PI_2    0.886226925452758014   /* sqrt(pi)/2         */
#define M_2_SQRT_2PI   5.013256549262001005   /* 2*sqrt(2*pi)       */
#define M_1_2_SQRT_2PI 0.199471140200716338   /* 1/(2*sqrt(2*pi))   */
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

#ifndef __APPLE__
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

typedef double ft_complex[2];

typedef struct {
    unsigned sse     : 1;
    unsigned sse2    : 1;
    unsigned sse3    : 1;
    unsigned ssse3   : 1;
    unsigned sse4_1  : 1;
    unsigned sse4_2  : 1;
    unsigned avx     : 1;
    unsigned avx2    : 1;
    unsigned fma     : 1;
    unsigned avx512f : 1;
} ft_simd;

static inline void cpuid(unsigned op, unsigned count, unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    #if defined(__i386__) && defined(__PIC__)
        __asm__ __volatile__
        ("mov %%ebx, %%edi;"
         "cpuid;"
         "xchgl %%ebx, %%edi;"
         : "=a" (*eax), "=D" (*ebx), "=c" (*ecx), "=d" (*edx) : "0" (op), "2" (count) : "cc");
    #else
        __asm__ __volatile__
        ("cpuid": "=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx) : "0" (op), "2" (count) : "cc");
    #endif
}

static inline ft_simd get_simd(void) {
    unsigned eax, ebx, ecx, edx;
    unsigned eax1, ebx1, ecx1, edx1;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    cpuid(7, 0, &eax1, &ebx1, &ecx1, &edx1);
    #ifndef bit_AVX512F
        return (ft_simd) {!!(edx & bit_SSE), !!(edx & bit_SSE2), !!(ecx & bit_SSE3), !!(ecx & bit_SSSE3), !!(ecx & bit_SSE4_1), !!(ecx & bit_SSE4_2), !!(ecx & bit_AVX), !!(ebx1 & bit_AVX2), !!(ecx & bit_FMA), 0};
    #else
        return (ft_simd) {!!(edx & bit_SSE), !!(edx & bit_SSE2), !!(ecx & bit_SSE3), !!(ecx & bit_SSSE3), !!(ecx & bit_SSE4_1), !!(ecx & bit_SSE4_2), !!(ecx & bit_AVX), !!(ebx1 & bit_AVX2), !!(ecx & bit_FMA), !!(ebx1 & bit_AVX512F)};
    #endif
}

#ifdef __AVX512F__
    #define VECTOR_SIZE_8 8
    #define ALIGN_SIZE VECTOR_SIZE_8
    typedef double double8 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
    #define vall8(x) ((double8) _mm512_set1_pd(x))
    #define vload8(v) ((double8) _mm512_load_pd(v))
    #define vloadu8(v) ((double8) _mm512_loadu_pd(v))
    #define vstore8(u, v) (_mm512_store_pd(u, v))
    #define vstoreu8(u, v) (_mm512_storeu_pd(u, v))
    #define vfma8(a, b, c) ((double8) _mm512_fmadd_pd(a, b, c))
    #define vfms8(a, b, c) ((double8) _mm512_fmsub_pd(a, b, c))
    #define vfmas8(a, b, c) ((double8) _mm512_fmaddsub_pd(a, b, c))
    typedef float float16 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
    #define vall16f(x) ((float16) _mm512_set1_ps(x))
    #define vload16f(v) ((float16) _mm512_load_ps(v))
    #define vloadu16f(v) ((float16) _mm512_loadu_ps(v))
    #define vstore16f(u, v) (_mm512_store_ps(u, v))
    #define vstoreu16f(u, v) (_mm512_storeu_ps(u, v))
    #define vfma16f(a, b, c) ((float16) _mm512_fmadd_ps(a, b, c))
    #define vfms16f(a, b, c) ((float16) _mm512_fmsub_ps(a, b, c))
    #define vfmas16f(a, b, c) ((float16) _mm512_fmaddsub_ps(a, b, c))
#endif
#ifdef __AVX__
    #define VECTOR_SIZE_4 4
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_4
    #endif
    typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
    #define vall4(x) ((double4) _mm256_set1_pd(x))
    #define vload4(v) ((double4) _mm256_load_pd(v))
    #define vloadu4(v) ((double4) _mm256_loadu_pd(v))
    #define vstore4(u, v) (_mm256_store_pd(u, v))
    #define vstoreu4(u, v) (_mm256_storeu_pd(u, v))
    typedef float float8 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
    #define vall8f(x) ((float8) _mm256_set1_ps(x))
    #define vload8f(v) ((float8) _mm256_load_ps(v))
    #define vloadu8f(v) ((float8) _mm256_loadu_ps(v))
    #define vstore8f(u, v) (_mm256_store_ps(u, v))
    #define vstoreu8f(u, v) (_mm256_storeu_ps(u, v))
    #define vas4(a, b) ((double4) _mm256_addsub_pd(a, b))
    #define vas8f(a, b) ((float8) _mm256_addsub_ps(a, b))
    #ifdef __FMA__
        #define vfma4(a, b, c) ((double4) _mm256_fmadd_pd(a, b, c))
        #define vfms4(a, b, c) ((double4) _mm256_fmsub_pd(a, b, c))
        #define vfmas4(a, b, c) ((double4) _mm256_fmaddsub_pd(a, b, c))
        #define vfma8f(a, b, c) ((float8) _mm256_fmadd_ps(a, b, c))
        #define vfms8f(a, b, c) ((float8) _mm256_fmsub_ps(a, b, c))
        #define vfmas8f(a, b, c) ((float8) _mm256_fmaddsub_ps(a, b, c))
    #endif
#endif
#ifdef __SSE2__
    #define VECTOR_SIZE_2 2
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_2
    #endif
    typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
    #define vall2(x) ((double2) _mm_set1_pd(x))
    #define vload2(v) ((double2) _mm_load_pd(v))
    #define vloadu2(v) ((double2) _mm_loadu_pd(v))
    #define vstore2(u, v) (_mm_store_pd(u, v))
    #define vstoreu2(u, v) (_mm_storeu_pd(u, v))
    #ifdef __FMA__
        #define vfma2(a, b, c) ((double2) _mm_fmadd_pd(a, b, c))
        #define vfms2(a, b, c) ((double2) _mm_fmsub_pd(a, b, c))
        #define vfmas2(a, b, c) ((double2) _mm_fmaddsub_pd(a, b, c))
    #endif
#endif
#ifdef __SSE__
    #define VECTOR_SIZE_2 2
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_2
    #endif
    typedef float float4 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
    #define vall4f(x) ((float4) _mm_set1_ps(x))
    #define vload4f(v) ((float4) _mm_load_ps(v))
    #define vloadu4f(v) ((float4) _mm_loadu_ps(v))
    #define vstore4f(u, v) (_mm_store_ps(u, v))
    #define vstoreu4f(u, v) (_mm_storeu_ps(u, v))
    #ifdef __FMA__
        #define vfma4f(a, b, c) ((float4) _mm_fmadd_ps(a, b, c))
        #define vfms4f(a, b, c) ((float4) _mm_fmsub_ps(a, b, c))
        #define vfmas4f(a, b, c) ((float4) _mm_fmaddsub_ps(a, b, c))
    #endif
#endif

#ifndef ALIGN_SIZE
    #define ALIGN_SIZE 1
#endif

#define VALIGN(N) N // ((N + ALIGN_SIZE - 1) & -ALIGN_SIZE)
#define VMALLOC(s) malloc(s) // _mm_malloc(s, ALIGN_SIZE*8)
#define VFREE(s) free(s) // _mm_free(s)

#define vmuladd(a, b, c) ((a)*(b)+(c))
#define vmulsub(a, b, c) ((a)*(b)-(c))

void horner_default(const int n, const double * c, const int incc, const int m, double * x, double * f);
void horner_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f);
void horner_AVX(const int n, const double * c, const int incc, const int m, double * x, double * f);
void horner_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f);
void horner_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f);

void horner_defaultf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void horner_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void horner_AVXf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void horner_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void horner_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f);

void clenshaw_default(const int n, const double * c, const int incc, const int m, double * x, double * f);
void clenshaw_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f);
void clenshaw_AVX(const int n, const double * c, const int incc, const int m, double * x, double * f);
void clenshaw_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f);
void clenshaw_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f);

void clenshaw_defaultf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void clenshaw_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void clenshaw_AVXf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void clenshaw_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f);
void clenshaw_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f);

void orthogonal_polynomial_clenshaw_default(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f);
void orthogonal_polynomial_clenshaw_SSE2(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f);
void orthogonal_polynomial_clenshaw_AVX(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f);
void orthogonal_polynomial_clenshaw_AVX_FMA(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f);
void orthogonal_polynomial_clenshaw_AVX512F(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f);

void orthogonal_polynomial_clenshaw_defaultf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f);
void orthogonal_polynomial_clenshaw_SSEf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f);
void orthogonal_polynomial_clenshaw_AVXf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f);
void orthogonal_polynomial_clenshaw_AVX_FMAf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f);
void orthogonal_polynomial_clenshaw_AVX512Ff(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f);

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
double * plan_associated_jacobi_to_jacobi(const int norm2, const int n, const int c, const double alpha, const double beta, const double gamma, const double delta);

typedef struct ft_rotation_plan_s {
    double * s;
    double * c;
    int n;
} ft_rotation_plan;

void kernel_sph_hi2lo_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_lo2hi_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_hi2lo_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_lo2hi_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_hi2lo_AVX(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_lo2hi_AVX(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_sph_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

void kernel_tri_hi2lo_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_lo2hi_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_hi2lo_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_lo2hi_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_hi2lo_AVX(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_lo2hi_AVX(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_tri_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

void kernel_disk_hi2lo_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_lo2hi_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_hi2lo_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_lo2hi_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_hi2lo_AVX(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_lo2hi_AVX(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
void kernel_disk_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

void kernel_tet_hi2lo_SSE2(const ft_rotation_plan * RP, const int L, const int m, double * A);
void kernel_tet_lo2hi_SSE2(const ft_rotation_plan * RP, const int L, const int m, double * A);
void kernel_tet_hi2lo_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A);
void kernel_tet_lo2hi_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A);
void kernel_tet_hi2lo_AVX512F(const ft_rotation_plan * RP, const int L, const int m, double * A);
void kernel_tet_lo2hi_AVX512F(const ft_rotation_plan * RP, const int L, const int m, double * A);

typedef struct ft_spin_rotation_plan_s {
    double * s1;
    double * c1;
    double * s2;
    double * c2;
    int n;
    int s;
} ft_spin_rotation_plan;

void kernel_spinsph_hi2lo_default(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_lo2hi_default(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_hi2lo_SSE2(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_lo2hi_SSE2(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_hi2lo_AVX(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_lo2hi_AVX(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_hi2lo_AVX_FMA(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
void kernel_spinsph_lo2hi_AVX_FMA(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);

void execute_sph_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_sph_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_sph_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sph_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);

void execute_sphv_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_sphv_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_sphv_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_sphv_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);

void execute_tri_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_tri_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_tri_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_tri_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);

void execute_disk_hi2lo_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_disk_lo2hi_default(const ft_rotation_plan * RP, double * A, const int M);
void execute_disk_hi2lo_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_SSE2(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_hi2lo_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_AVX_FMA(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_hi2lo_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);
void execute_disk_lo2hi_AVX512F(const ft_rotation_plan * RP, double * A, double * B, const int M);

void execute_tet_hi2lo_SSE2(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void execute_tet_lo2hi_SSE2(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void execute_tet_hi2lo_AVX(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void execute_tet_lo2hi_AVX(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void execute_tet_hi2lo_AVX512F(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void execute_tet_lo2hi_AVX512F(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);

void execute_spinsph_hi2lo_default(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M);
void execute_spinsph_lo2hi_default(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M);
void execute_spinsph_hi2lo_SSE2(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M);
void execute_spinsph_lo2hi_SSE2(const ft_spin_rotation_plan * SRP, ft_complex * A, const int M);
void execute_spinsph_hi2lo_AVX(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);
void execute_spinsph_lo2hi_AVX(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);
void execute_spinsph_hi2lo_AVX_FMA(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);
void execute_spinsph_lo2hi_AVX_FMA(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);


void permute(const double * A, double * B, const int N, const int M, const int L);
void permute_t(double * A, const double * B, const int N, const int M, const int L);

void permute_sph(const double * A, double * B, const int N, const int M, const int L);
void permute_t_sph(double * A, const double * B, const int N, const int M, const int L);

void permute_tri(const double * A, double * B, const int N, const int M, const int L);
void permute_t_tri(double * A, const double * B, const int N, const int M, const int L);

void old_permute_tri(const double * A, double * B, const int N, const int M, const int L);
void old_permute_t_tri(double * A, const double * B, const int N, const int M, const int L);

#define permute_disk(A, B, N, M, L) permute_sph(A, B, N, M, L)
#define permute_t_disk(A, B, N, M, L) permute_t_sph(A, B, N, M, L)

void swap_warp(double * A, double * B, const int N);
void swap_warp_default(double * A, double * B, const int N);
void swap_warp_SSE2(double * A, double * B, const int N);
void swap_warp_AVX(double * A, double * B, const int N);
void swap_warp_AVX512F(double * A, double * B, const int N);

void swap_warpf(float * A, float * B, const int N);
void swap_warp_defaultf(float * A, float * B, const int N);
void swap_warp_SSEf(float * A, float * B, const int N);
void swap_warp_AVXf(float * A, float * B, const int N);
void swap_warp_AVX512Ff(float * A, float * B, const int N);

void warp(double * A, const int N, const int M, const int L);
void warp_t(double * A, const int N, const int M, const int L);

// A bitwise OR ('|') of zero or more of the following: FFTW_ESTIMATE FFTW_MEASURE FFTW_PATIENT FFTW_EXHAUSTIVE FFTW_WISDOM_ONLY FFTW_DESTROY_INPUT FFTW_PRESERVE_INPUT FFTW_UNALIGNED
#define FT_FFTW_FLAGS FFTW_MEASURE | FFTW_DESTROY_INPUT

#endif // FTINTERNAL_H
