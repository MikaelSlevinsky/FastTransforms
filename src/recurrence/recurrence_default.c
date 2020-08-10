#include "recurrence.h"

void horner_default(const int n, const double * c, const int incc, const int m, double * x, double * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0;
        return;
    }
    for (int j = 0; j < m; j++) {
        double bk = 0.0;
        for (int k = n-1; k >= 0; k--)
            bk = x[j]*bk + c[k*incc];
        f[j] = bk;
    }
}

void horner_defaultf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0f;
        return;
    }
    for (int j = 0; j < m; j++) {
        float bk = 0.0f;
        for (int k = n-1; k >= 0; k--)
            bk = x[j]*bk + c[k*incc];
        f[j] = bk;
    }
}

void clenshaw_default(const int n, const double * c, const int incc, const int m, double * x, double * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0;
        return;
    }
    for (int j = 0; j < m; j++) {
        double bk = 0.0;
        double bk1 = 0.0;
        double bk2 = 0.0;
        x[j] *= 2.0;
        for (int k = n-1; k >= 1; k--) {
            bk = x[j]*bk1 + c[k*incc] - bk2;
            bk2 = bk1;
            bk1 = bk;
        }
        x[j] *= 0.5;
        f[j] = x[j]*bk1 + c[0] - bk2;
    }
}

void clenshaw_defaultf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0f;
        return;
    }
    for (int j = 0; j < m; j++) {
        float bk = 0.0f;
        float bk1 = 0.0f;
        float bk2 = 0.0f;
        x[j] *= 2.0f;
        for (int k = n-1; k >= 1; k--) {
            bk = x[j]*bk1 + c[k*incc] - bk2;
            bk2 = bk1;
            bk1 = bk;
        }
        x[j] *= 0.5f;
        f[j] = x[j]*bk1 + c[0] - bk2;
    }
}

void orthogonal_polynomial_clenshaw_default(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0;
        return;
    }
    for (int j = 0; j < m; j++) {
        double bk = 0.0;
        double bk1 = 0.0;
        double bk2 = 0.0;
        double X = x[j];
        for (int k = n-1; k >= 0; k--) {
            bk = (A[k]*X+B[k])*bk1 - C[k+1]*bk2 + c[k*incc];
            bk2 = bk1;
            bk1 = bk;
        }
        f[j] = phi0[j]*bk;
    }
}

void orthogonal_polynomial_clenshaw_defaultf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0f;
        return;
    }
    for (int j = 0; j < m; j++) {
        float bk = 0.0f;
        float bk1 = 0.0f;
        float bk2 = 0.0f;
        float X = x[j];
        for (int k = n-1; k >= 0; k--) {
            bk = (A[k]*X+B[k])*bk1 - C[k+1]*bk2 + c[k*incc];
            bk2 = bk1;
            bk1 = bk;
        }
        f[j] = phi0[j]*bk;
    }
}

void orthogonal_polynomial_clenshaw_defaultl(const int n, const long double * c, const int incc, const long double * A, const long double * B, const long double * C, const int m, long double * x, long double * phi0, long double * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0l;
        return;
    }
    for (int j = 0; j < m; j++) {
        long double bk = 0.0l;
        long double bk1 = 0.0l;
        long double bk2 = 0.0l;
        long double X = x[j];
        for (int k = n-1; k >= 0; k--) {
            bk = (A[k]*X+B[k])*bk1 - C[k+1]*bk2 + c[k*incc];
            bk2 = bk1;
            bk1 = bk;
        }
        f[j] = phi0[j]*bk;
    }
}

#if defined(FT_QUADMATH)
    void orthogonal_polynomial_clenshaw_defaultq(const int n, const quadruple * c, const int incc, const quadruple * A, const quadruple * B, const quadruple * C, const int m, quadruple * x, quadruple * phi0, quadruple * f) {
        if (n < 1) {
            for (int j = 0; j < m; j++)
                f[j] = 0.0q;
            return;
        }
        for (int j = 0; j < m; j++) {
            quadruple bk = 0.0q;
            quadruple bk1 = 0.0q;
            quadruple bk2 = 0.0q;
            quadruple X = x[j];
            for (int k = n-1; k >= 0; k--) {
                bk = (A[k]*X+B[k])*bk1 - C[k+1]*bk2 + c[k*incc];
                bk2 = bk1;
                bk1 = bk;
            }
            f[j] = phi0[j]*bk;
        }
    }
#endif
