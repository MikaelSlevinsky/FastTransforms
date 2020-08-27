#include "recurrence.h"

#ifdef __AVX512F__
    void horner_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        HORNER_KERNEL(double, double8, 8, 8, vloadu8, vstoreu8, vfma8, vall8)
    }
    void horner_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        HORNER_KERNEL(float, float16, 16, 8, vloadu16f, vstoreu16f, vfma16f, vall16f)
    }
    void clenshaw_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        CLENSHAW_KERNEL(double, double8, 8, 4, vloadu8, vstoreu8, vfma8, vall8)
    }
    void clenshaw_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        CLENSHAW_KERNEL(float, float16, 16, 4, vloadu16f, vstoreu16f, vfma16f, vall16f)
    }
    void orthogonal_polynomial_clenshaw_AVX512F(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {
        ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(double, double8, 8, 4, vloadu8, vstoreu8, vfma8, vfms8, vall8)
    }
    void orthogonal_polynomial_clenshaw_AVX512Ff(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {
        ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(float, float16, 16, 4, vloadu16f, vstoreu16f, vfma16f, vfms16f, vall16f)
    }
    void eigen_eval_AVX512F(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, const int sign, double * f) {
        EIGEN_EVAL_KERNEL(double, double8, 8, 4, vloadu8, vstoreu8, vfma8, vfms8, vall8, vsqrt8, vmovemask8, eps()/floatmin(), sqrt)
    }
    void eigen_eval_AVX512Ff(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, const int sign, float * f) {
        EIGEN_EVAL_KERNEL(float, float16, 16, 4, vloadu16f, vstoreu16f, vfma16f, vfms16f, vall16f, vsqrt16f, vmovemask16f, epsf()/floatminf(), sqrtf)
    }
#else
    void horner_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_AVX_FMA(n, c, incc, m, x, f);}
    void horner_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_AVX_FMAf(n, c, incc, m, x, f);}
    void clenshaw_AVX512F(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_AVX_FMA(n, c, incc, m, x, f);}
    void clenshaw_AVX512Ff(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_AVX_FMAf(n, c, incc, m, x, f);}
    void orthogonal_polynomial_clenshaw_AVX512F(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {orthogonal_polynomial_clenshaw_AVX_FMA(n, c, incc, A, B, C, m, x, phi0, f);}
    void orthogonal_polynomial_clenshaw_AVX512Ff(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {orthogonal_polynomial_clenshaw_AVX_FMAf(n, c, incc, A, B, C, m, x, phi0, f);}
    void eigen_eval_AVX512F(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, const int sign, double * f) {eigen_eval_AVX_FMA(n, c, incc, A, B, C, m, x, sign, f);}
    void eigen_eval_AVX512Ff(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, const int sign, float * f) {eigen_eval_AVX_FMAf(n, c, incc, A, B, C, m, x, sign, f);}
#endif
