#include "recurrence.h"

#ifdef __SSE2__
    void horner_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        HORNER_KERNEL(double, double2, 2, 16, vloadu2, vstoreu2, vmuladd, vall2)
    }
    void clenshaw_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        CLENSHAW_KERNEL(double, double2, 2, 8, vloadu2, vstoreu2, vmuladd)
    }
    void orthogonal_polynomial_clenshaw_SSE2(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {
        ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(double, double2, 2, 8, vloadu2, vstoreu2, vmuladd, vmulsub, vall2)
    }
#else
    void horner_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_default(n, c, incc, m, x, f);}
    void clenshaw_SSE2(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_default(n, c, incc, m, x, f);}
    void orthogonal_polynomial_clenshaw_SSE2(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {orthogonal_polynomial_clenshaw_default(n, c, incc, A, B, C, m, x, phi0, f);}
#endif
