#include "recurrence.h"

#ifdef __SSE__
    void horner_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        HORNER_KERNEL(float, float4, 4, 16, vloadu4f, vstoreu4f, vmuladd, vall4f)
    }
    void clenshaw_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        CLENSHAW_KERNEL(float, float4, 4, 8, vloadu4f, vstoreu4f, vmuladd, vall4f)
    }
    void orthogonal_polynomial_clenshaw_SSEf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {
        ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(float, float4, 4, 8, vloadu4f, vstoreu4f, vmuladd, vmulsub, vall4f)
    }
    void eigen_eval_SSEf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, const int sign, float * f) {
        EIGEN_EVAL_KERNEL(float, float4, 4, 8, vloadu4f, vstoreu4f, vmuladd, vall4f, vsqrt4f, vmovemask4f, epsf()/floatminf(), sqrtf)
    }
#else
    void horner_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_defaultf(n, c, incc, m, x, f);}
    void clenshaw_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_defaultf(n, c, incc, m, x, f);}
    void orthogonal_polynomial_clenshaw_SSEf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {orthogonal_polynomial_clenshaw_defaultf(n, c, incc, A, B, C, m, x, phi0, f);}
    void eigen_eval_SSEf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, const int sign, float * f) {eigen_eval_defaultf(n, c, incc, A, B, C, m, x, sign, f);}
#endif
