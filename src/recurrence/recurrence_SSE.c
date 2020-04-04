#include "ftinternal.h"

#ifdef __SSE__
    void horner_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        HORNER_KERNEL(float, float4, 4, 16, vloadu4f, vstoreu4f, vmuladd, vall4f)
    }
    void clenshaw_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        CLENSHAW_KERNEL(float, float4, 4, 8, vloadu4f, vstoreu4f, vmuladd)
    }
#else
    void horner_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_defaultf(n, c, incc, m, x, f);}
    void clenshaw_SSEf(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_defaultf(n, c, incc, m, x, f);}
#endif
