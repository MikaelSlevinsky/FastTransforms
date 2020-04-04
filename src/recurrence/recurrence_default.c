#include "ftinternal.h"

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
