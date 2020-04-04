#include "ftinternal.h"

#ifdef __AVX__
    void horner_AVX(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        HORNER_KERNEL(double, double4, 4, 8, vloadu4, vstoreu4, vmuladd, vall4)
    }
    void horner_AVXf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        HORNER_KERNEL(float, float8, 8, 8, vloadu8f, vstoreu8f, vmuladd, vall8f)
    }
    void clenshaw_AVX(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        CLENSHAW_KERNEL(double, double4, 4, 4, vloadu4, vstoreu4, vmuladd)
    }
    void clenshaw_AVXf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        CLENSHAW_KERNEL(float, float8, 8, 4, vloadu8f, vstoreu8f, vmuladd)
    }
#else
    void horner_AVX(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_SSE2(n, c, incc, m, x, f);}
    void horner_AVXf(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_SSEf(n, c, incc, m, x, f);}
    void clenshaw_AVX(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_SSE2(n, c, incc, m, x, f);}
    void clenshaw_AVXf(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_SSEf(n, c, incc, m, x, f);}
#endif
