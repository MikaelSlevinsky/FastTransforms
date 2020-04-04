#include "ftinternal.h"

#ifdef __AVX512F__
    void horner_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        HORNER_KERNEL(double, double8, 8, 4, vloadu8, vstoreu8, vfma8, vall8)
    }
    void horner_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        HORNER_KERNEL(float, float16, 16, 4, vloadu16f, vstoreu16f, vfma16f, vall16f)
    }
    void clenshaw_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        CLENSHAW_KERNEL(double, double8, 8, 2, vloadu8, vstoreu8, vfma8)
    }
    void clenshaw_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        CLENSHAW_KERNEL(float, float16, 16, 2, vloadu16f, vstoreu16f, vfma16f)
    }
#else
    void horner_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_AVX_FMA(n, c, incc, m, x, f);}
    void horner_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_AVX_FMAf(n, c, incc, m, x, f);}
    void clenshaw_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_AVX_FMA(n, c, incc, m, x, f);}
    void clenshaw_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_AVX_FMAf(n, c, incc, m, x, f);}
#endif