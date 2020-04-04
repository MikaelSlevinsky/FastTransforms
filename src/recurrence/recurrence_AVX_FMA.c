#include "ftinternal.h"

#ifdef __AVX__
    #ifdef __FMA__
        void horner_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f) {
            HORNER_KERNEL(double, double4, 4, 4, vloadu4, vstoreu4, vfma4, vall4)
        }
        void horner_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
            HORNER_KERNEL(float, float8, 8, 4, vloadu8f, vstoreu8f, vfma8f, vall8f)
        }
        void clenshaw_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f) {
            CLENSHAW_KERNEL(double, double4, 4, 2, vloadu4, vstoreu4, vfma4)
        }
        void clenshaw_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
            CLENSHAW_KERNEL(float, float8, 8, 2, vloadu8f, vstoreu8f, vfma8f)
        }
    #else
        void horner_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_AVX(n, c, incc, m, x, f);}
        void horner_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_AVXf(n, c, incc, m, x, f);}
        void clenshaw_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_AVX(n, c, incc, m, x, f);}
        void clenshaw_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_AVXf(n, c, incc, m, x, f);}
    #endif
#else
    void horner_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_SSE2(n, c, incc, m, x, f);}
    void horner_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_SSEf(n, c, incc, m, x, f);}
    void clenshaw_AVX_FMA(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_SSE2(n, c, incc, m, x, f);}
    void clenshaw_AVX_FMAf(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_SSEf(n, c, incc, m, x, f);}
#endif
