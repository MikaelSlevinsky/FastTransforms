// Computational routines for the harmonic polynomial connection problem.

#include "rotations.h"

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

RotationPlan * plan_rotsphere(const int n) {
    double * s = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double * c = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int l = 0; l < n; l++)
        for (int m = 0; m < n-l; m++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2)*(2*l+2*m+5);
            den = (l+2*m+3)*(l+2*m+4);
            s(l, m) = sqrt(nums/den);
            c(l, m) = sqrt(numc/den);
        }
    RotationPlan * RP = malloc(sizeof(RotationPlan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

// Convert a single vector of spherical harmonics of order m to 0/1.

void kernel1_sph_hi2lo(const RotationPlan * RP, double * A, const int m) {
    double * s = RP->s, * c = RP->c;
    int n = RP->n;
    double ss, cc, a1, a2;
    for (int j = m; j > 1; j -= 2) {
        for (int l = n-1-j; l >= 0; l--) {
            ss = s(l, j-2);
            cc = c(l, j-2);
            a1 = A[l];
            a2 = A[l+2];
            A[l] = cc*a1 + ss*a2;
            A[l+2] = cc*a2 - ss*a1;
        }
    }
}

// Convert a single vector of spherical harmonics of order 0/1 to m.

void kernel1_sph_lo2hi(const RotationPlan * RP, double * A, const int m) {
    double * s = RP->s, * c = RP->c;
    int n = RP->n;
    double ss, cc, a1, a2;
    for (int j = 2+m%2; j <= m; j += 2) {
        for (int l = 0; l <= n-1-j; l++) {
            ss = s(l, j-2);
            cc = c(l, j-2);
            a1 = A[l];
            a2 = A[l+2];
            A[l] = cc*a1 - ss*a2;
            A[l+2] = cc*a2 + ss*a1;
        }
    }
}

// Convert a pair of vectors of spherical harmonics of order m to 0/1.

void kernel2_sph_hi2lo(const RotationPlan * RP, double * A, double * B, const int m) {
    double * s = RP->s, * c = RP->c;
    int n = RP->n;
    double ss, cc, a1, a2, a3, a4;
    for (int j = m; j > 1; j -= 2) {
        for (int l = n-1-j; l >= 0; l--) {
            ss = s(l, j-2);
            cc = c(l, j-2);
            a1 = A[l];
            a2 = A[l+2];
            a3 = B[l];
            a4 = B[l+2];
            A[l] = cc*a1 + ss*a2;
            A[l+2] = cc*a2 - ss*a1;
            B[l] = cc*a3 + ss*a4;
            B[l+2] = cc*a4 - ss*a3;
        }
    }
}

// Convert a pair of vectors of spherical harmonics of order 0/1 to m.

void kernel2_sph_lo2hi(const RotationPlan * RP, double * A, double * B, const int m) {
    double * s = RP->s, * c = RP->c;
    int n = RP->n;
    double ss, cc, a1, a2, a3, a4;
    for (int j = 2+m%2; j <= m; j += 2) {
        for (int l = 0; l <= n-1-j; l++) {
            ss = s(l, j-2);
            cc = c(l, j-2);
            a1 = A[l];
            a2 = A[l+2];
            a3 = B[l];
            a4 = B[l+2];
            A[l] = cc*a1 - ss*a2;
            A[l+2] = cc*a2 + ss*a1;
            B[l] = cc*a3 - ss*a4;
            B[l+2] = cc*a4 + ss*a3;
        }
    }
}
