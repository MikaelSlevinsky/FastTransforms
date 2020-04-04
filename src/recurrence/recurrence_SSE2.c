#include "ftinternal.h"

#ifdef __SSE2__
    void horner_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        HORNER_KERNEL(double, double2, 2, 16, vloadu2, vstoreu2, vmuladd, vall2)
    }
    void clenshaw_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        CLENSHAW_KERNEL(double, double2, 2, 8, vloadu2, vstoreu2, vmuladd)
    }
#else
    void horner_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_default(n, c, incc, m, x, f);}
    void clenshaw_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_default(n, c, incc, m, x, f);}
#endif
