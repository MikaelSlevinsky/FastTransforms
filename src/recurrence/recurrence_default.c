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

void eigen_eval_default(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, const int sign, double * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0;
        return;
    }
    for (int j = 0; j < m; j++) {
        double vkm1 = 1.0;
        double vk = 1.0;
        double vkp1 = 0.0;
        double nrm = 1.0;
        double X = x[j];
        f[j] = c[(n-1)*incc];
        for (int k = n-1; k > 0; k--) {
            vkm1 = (A[k]*X+B[k])*vk - C[k]*vkp1;
            vkp1 = vk;
            vk = vkm1;
            nrm += vkm1*vkm1;
            f[j] += vkm1*c[(k-1)*incc];
            if (nrm > eps()/floatmin()) {
                nrm = 1.0/sqrt(nrm);
                vkp1 *= nrm;
                vk *= nrm;
                vkm1 *= nrm;
                f[j] *= nrm;
                nrm = 1.0;
            }
        }
        nrm = (sign*vkm1 < 0) ? -1.0/sqrt(nrm) : 1.0/sqrt(nrm);
        f[j] *= nrm;
    }
}

void eigen_eval_defaultf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, const int sign, float * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0f;
        return;
    }
    for (int j = 0; j < m; j++) {
        float vkm1 = 1.0f;
        float vk = 1.0f;
        float vkp1 = 0.0f;
        float nrm = 1.0f;
        float X = x[j];
        f[j] = c[(n-1)*incc];
        for (int k = n-1; k > 0; k--) {
            vkm1 = (A[k]*X+B[k])*vk - C[k]*vkp1;
            vkp1 = vk;
            vk = vkm1;
            nrm += vkm1*vkm1;
            f[j] += vkm1*c[(k-1)*incc];
            if (nrm > epsf()/floatminf()) {
                nrm = 1.0f/sqrtf(nrm);
                vkp1 *= nrm;
                vk *= nrm;
                vkm1 *= nrm;
                f[j] *= nrm;
                nrm = 1.0f;
            }
        }
        nrm = (sign*vkm1 < 0) ? -1.0f/sqrtf(nrm) : 1.0f/sqrtf(nrm);
        f[j] *= nrm;
    }
}

void eigen_eval_defaultl(const int n, const long double * c, const int incc, const long double * A, const long double * B, const long double * C, const int m, long double * x, const int sign, long double * f) {
    if (n < 1) {
        for (int j = 0; j < m; j++)
            f[j] = 0.0l;
        return;
    }
    for (int j = 0; j < m; j++) {
        long double vkm1 = 1.0l;
        long double vk = 1.0l;
        long double vkp1 = 0.0l;
        long double nrm = 1.0l;
        long double X = x[j];
        f[j] = c[(n-1)*incc];
        for (int k = n-1; k > 0; k--) {
            vkm1 = (A[k]*X+B[k])*vk - C[k]*vkp1;
            vkp1 = vk;
            vk = vkm1;
            nrm += vkm1*vkm1;
            f[j] += vkm1*c[(k-1)*incc];
            if (nrm > epsl()/floatminl()) {
                nrm = 1.0l/sqrtl(nrm);
                vkp1 *= nrm;
                vk *= nrm;
                vkm1 *= nrm;
                f[j] *= nrm;
                nrm = 1.0l;
            }
        }
        nrm = (sign*vkm1 < 0) ? -1.0l/sqrtl(nrm) : 1.0l/sqrtl(nrm);
        f[j] *= nrm;
    }
}

#if defined(FT_QUADMATH)
    void eigen_eval_defaultq(const int n, const quadruple * c, const int incc, const quadruple * A, const quadruple * B, const quadruple * C, const int m, quadruple * x, const int sign, quadruple * f) {
        if (n < 1) {
            for (int j = 0; j < m; j++)
                f[j] = 0.0q;
            return;
        }
        for (int j = 0; j < m; j++) {
            quadruple vkm1 = 1.0q;
            quadruple vk = 1.0q;
            quadruple vkp1 = 0.0q;
            quadruple nrm = 1.0q;
            quadruple X = x[j];
            f[j] = c[(n-1)*incc];
            for (int k = n-1; k > 0; k--) {
                vkm1 = (A[k]*X+B[k])*vk - C[k]*vkp1;
                vkp1 = vk;
                vk = vkm1;
                nrm += vkm1*vkm1;
                f[j] += vkm1*c[(k-1)*incc];
                if (nrm > epsq()/floatminq()) {
                    nrm = 1.0q/sqrtq(nrm);
                    vkp1 *= nrm;
                    vk *= nrm;
                    vkm1 *= nrm;
                    f[j] *= nrm;
                    nrm = 1.0q;
                }
            }
            nrm = (sign*vkm1 < 0) ? -1.0q/sqrtq(nrm) : 1.0q/sqrtq(nrm);
            f[j] *= nrm;
        }
    }
#endif
