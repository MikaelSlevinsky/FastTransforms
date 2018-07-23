// Staging ground for determining the minimal assembly Givens instructions.

#include <immintrin.h>

#if __SSE2__
    typedef float float4 __attribute__ ((vector_size (4*4)));
    #define vsetf4(x) ((float4) _mm_set1_ps(x))
    #define vloadf4(v) ((float4) _mm_loadu_ps(v))
    #define vstoref4(u, v) (_mm_storeu_ps(u, v))
    typedef double double2 __attribute__ ((vector_size (2*8)));
    #define vset2(x) ((double2) _mm_set1_pd(x))
    #define vload2(v) ((double2) _mm_loadu_pd(v))
    #define vstore2(u, v) (_mm_storeu_pd(u, v))
#endif
#if __AVX__
    typedef float float8 __attribute__ ((vector_size (8*4)));
    #define vsetf8(x) ((float8) _mm256_set1_ps(x))
    #define vloadf8(v) ((float8) _mm256_loadu_ps(v))
    #define vstoref8(u, v) (_mm256_storeu_ps(u, v))
    typedef double double4 __attribute__ ((vector_size (4*8)));
    #define vset4(x) ((double4) _mm256_set1_pd(x))
    #define vload4(v) ((double4) _mm256_loadu_pd(v))
    #define vstore4(u, v) (_mm256_storeu_pd(u, v))
#endif
#if __AVX512F__
    typedef float float16 __attribute__ ((vector_size (16*4)));
    #define vsetf16(x) ((float16) _mm512_set1_ps(x))
    #define vloadf16(v) ((float16) _mm512_loadu_ps(v))
    #define vstoref16(u, v) (_mm512_storeu_ps(u, v))
    typedef double double8 __attribute__ ((vector_size (8*8)));
    #define vset8(x) ((double8) _mm512_set1_pd(x))
    #define vload8(v) ((double8) _mm512_loadu_pd(v))
    #define vstore8(u, v) (_mm512_storeu_pd(u, v))
#endif


void apply_givensf(const float S, const float C, float * X, float * Y) {
    float x = C*X[0] + S*Y[0];
    float y = C*Y[0] - S*X[0];

    X[0] = x;
    Y[0] = y;
}
void apply_givens(const double S, const double C, double * X, double * Y) {
    double x = C*X[0] + S*Y[0];
    double y = C*Y[0] - S*X[0];

    X[0] = x;
    Y[0] = y;
}

#if __SSE2__
    void apply_givensf_SSE(const float S, const float C, float * X, float * Y) {
        float4 x = vloadf4(X);
        float4 y = vloadf4(Y);

        vstoref4(X, C*x + S*y);
        vstoref4(Y, C*y - S*x);
    }
    void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, C*x + S*y);
        vstore2(Y, C*y - S*x);
    }
#endif

#if __AVX__
    void apply_givensf_AVX(const float S, const float C, float * X, float * Y) {
        float8 x = vloadf8(X);
        float8 y = vloadf8(Y);

        vstoref8(X, C*x + S*y);
        vstoref8(Y, C*y - S*x);
    }
    void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, C*x + S*y);
        vstore4(Y, C*y - S*x);
    }
#endif

#if __AVX512F__
    void apply_givensf_AVX512(const float S, const float C, float * X, float * Y) {
        float16 x = vloadf16(X);
        float16 y = vloadf16(Y);

        vstoref16(X, C*x + S*y);
        vstoref16(Y, C*y - S*x);
    }
    void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, C*x + S*y);
        vstore8(Y, C*y - S*x);
    }
#endif
