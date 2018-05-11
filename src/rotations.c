// Computational routines for the harmonic polynomial connection problem.

#include "fasttransforms.h"

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

RotationPlan * plan_rotsphere(const int n) {
    double * s = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double * c = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++) {
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

RotationPlan * plan_rottriangle(const int n, const double alpha, const double beta, const double gamma) {
    double * s = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double * c = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+alpha+1);
            numc = (2*m+beta+gamma+2)*(2*l+2*m+alpha+beta+gamma+4);
            den = (l+2*m+beta+gamma+3)*(l+2*m+alpha+beta+gamma+3);
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

void kernel_sph_hi2lo(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens(2, RP->s, RP->c, n, l, j, A);
}

// Convert a single vector of spherical harmonics of order 0/1 to m.

void kernel_sph_lo2hi(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t(2, RP->s, RP->c, n, l, j, A);
}

// Convert a pair of vectors of spherical harmonics of order m to 0/1.
// The pair of vectors are stored in A in row-major ordering.

void kernel_sph_hi2lo_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_SSE(2, RP->s, RP->c, n, l, j, A);
}

// Convert a pair of vectors of spherical harmonics of order 0/1 to m.
// The pair of vectors are stored in A in row-major ordering.

void kernel_sph_lo2hi_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_SSE(2, RP->s, RP->c, n, l, j, A);
}

// Convert a single vector of triangular harmonics of order m to 0.

void kernel_tri_hi2lo(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-1; j >= 0; j -= 1)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens(1, RP->s, RP->c, n, l, j, A);
}

// Convert a single vector of triangular harmonics of order 0 to m.

void kernel_tri_lo2hi(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = 0; j < m; j += 1)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t(1, RP->s, RP->c, n, l, j, A);
}


static inline void apply_givens(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A) {
    double s1 = s(l, m);
    double c1 = c(l, m);

    double a1 = A[l];
    double a2 = A[l+inc];

    A[l    ] = c1*a1 + s1*a2;
    A[l+inc] = c1*a2 - s1*a1;
}

static inline void apply_givens_t(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A) {
    double s1 = s(l, m);
    double c1 = c(l, m);

    double a1 = A[l];
    double a2 = A[l+inc];

    A[l    ] = c1*a1 - s1*a2;
    A[l+inc] = c1*a2 + s1*a1;
}

static inline void apply_givens_SSE(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A) {
    double2 s1 = vall2(s(l, m));
    double2 c1 = vall2(c(l, m));

    double2 a1 = vload2(A+2*l      );
    double2 a2 = vload2(A+2*(l+inc));

    vstore2(A+2*l,       c1*a1 + s1*a2);
    vstore2(A+2*(l+inc), c1*a2 - s1*a1);
}

static inline void apply_givens_t_SSE(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A) {
    double2 s1 = vall2(s(l, m));
    double2 c1 = vall2(c(l, m));

    double2 a1 = vload2(A+2*l      );
    double2 a2 = vload2(A+2*(l+inc));

    vstore2(A+2*l,       c1*a1 - s1*a2);
    vstore2(A+2*(l+inc), c1*a2 + s1*a1);
}

static inline void apply_givens_AVX(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A) {
    double4 s1 = vall4(s(l, m));
    double4 c1 = vall4(c(l, m));

    double4 a1 = vload4(A+4*l      );
    double4 a2 = vload4(A+4*(l+inc));

    vstore4(A+4*l,       c1*a1 + s1*a2);
    vstore4(A+4*(l+inc), c1*a2 - s1*a1);
}

static inline void apply_givens_t_AVX(const int inc, const double * s, const double * c, const int n, const int l, const int m, double * A) {
    double4 s1 = vall4(s(l, m));
    double4 c1 = vall4(c(l, m));

    double4 a1 = vload4(A+4*l      );
    double4 a2 = vload4(A+4*(l+inc));

    vstore4(A+4*l,       c1*a1 - s1*a2);
    vstore4(A+4*(l+inc), c1*a2 + s1*a1);
}
