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
            apply_givens(RP->s(l, j), RP->c(l, j), A+l, A+l+2);
}

// Convert a single vector of spherical harmonics of order 0/1 to m.

void kernel_sph_lo2hi(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t(RP->s(l, j), RP->c(l, j), A+l, A+l+2);
}

// Convert a pair of vectors of spherical harmonics of order m to 0/1.
// The pair of vectors are stored in A in row-major ordering.

void kernel_sph_hi2lo_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+2));
}

// Convert a pair of vectors of spherical harmonics of order 0/1 to m.
// The pair of vectors are stored in A in row-major ordering.

void kernel_sph_lo2hi_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+2));
}

// Convert four vectors of spherical harmonics of order m, m, m+2, m+2 to 0/1.
// The four vectors are stored in A in row-major ordering.

void kernel_sph_hi2lo_AVX(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-3-m; l >= 0; l--)
        apply_givens_SSE(RP->s(l, m), RP->c(l, m), A+4*l+2, A+4*(l+2)+2);
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_AVX(RP->s(l, j), RP->c(l, j), A+4*l, A+4*(l+2));
}

// Convert four vectors of spherical harmonics of order 0/1 to m, m, m+2, m+2.
// The four vectors are stored in A in row-major ordering.

void kernel_sph_lo2hi_AVX(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_AVX(RP->s(l, j), RP->c(l, j), A+4*l, A+4*(l+2));
    for (int l = 0; l <= n-3-m; l++)
        apply_givens_t_SSE(RP->s(l, m), RP->c(l, m), A+4*l+2, A+4*(l+2)+2);
}


// Convert a single vector of triangular harmonics of order m to 0.

void kernel_tri_hi2lo(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-1; j >= 0; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens(RP->s(l, j), RP->c(l, j), A+l, A+l+1);
}

// Convert a single vector of triangular harmonics of order 0 to m.

void kernel_tri_lo2hi(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = 0; j < m; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t(RP->s(l, j), RP->c(l, j), A+l, A+l+1);
}

// Convert two vectors of triangular harmonics of order m and m+1 to 0.

void kernel_tri_hi2lo_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-2-m; l >= 0; l--)
        apply_givens(RP->s(l, m), RP->c(l, m), A+2*l+1, A+2*(l+1)+1);
    for (int j = m-1; j >= 0; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+1));
}

// Convert two vectors of triangular harmonics of order 0 to m and m+1.

void kernel_tri_lo2hi_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = 0; j < m; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+1));
    for (int l = 0; l <= n-2-m; l++)
        apply_givens_t(RP->s(l, m), RP->c(l, m), A+2*l+1, A+2*(l+1)+1);
}

// Convert four vectors of triangular harmonics of order m, m+1, m+2, m+3 to 0.

void kernel_tri_hi2lo_AVX(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-2-m; l >= 0; l--)
        apply_givens(RP->s(l, m), RP->c(l, m), A+4*l+1, A+4*(l+1)+1);
    for (int l = n-4-m; l >= 0; l--)
        apply_givens(RP->s(l, m+2), RP->c(l, m+2), A+4*l+3, A+4*(l+1)+3);
    for (int j = m+1; j >= m; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+4*l+2, A+4*(l+1)+2);
    for (int j = m-1; j >= 0; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_AVX(RP->s(l, j), RP->c(l, j), A+4*l, A+4*(l+1));
}

// Convert four vectors of triangular harmonics of order 0 to m, m+1, m+2, m+3.

void kernel_tri_lo2hi_AVX(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = 0; j < m; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_AVX(RP->s(l, j), RP->c(l, j), A+4*l, A+4*(l+1));
    for (int j = m; j <= m+1; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+4*l+2, A+4*(l+1)+2);
    for (int l = 0; l <= n-4-m; l++)
        apply_givens_t(RP->s(l, m+2), RP->c(l, m+2), A+4*l+3, A+4*(l+1)+3);
    for (int l = 0; l <= n-2-m; l++)
        apply_givens_t(RP->s(l, m), RP->c(l, m), A+4*l+1, A+4*(l+1)+1);
}

// Convert four vectors of triangular harmonics of order m, m+1, m+2, m+3, m+4, m+5, m+6, m+7 to 0.

void kernel_tri_hi2lo_AVX512(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-2-m; l >= 0; l--)
        apply_givens(RP->s(l, m), RP->c(l, m), A+8*l+1, A+8*(l+1)+1);
    for (int l = n-4-m; l >= 0; l--)
        apply_givens(RP->s(l, m+2), RP->c(l, m+2), A+8*l+3, A+8*(l+1)+3);
    for (int l = n-6-m; l >= 0; l--)
        apply_givens(RP->s(l, m+4), RP->c(l, m+4), A+8*l+5, A+8*(l+1)+5);
    for (int l = n-8-m; l >= 0; l--)
        apply_givens(RP->s(l, m+6), RP->c(l, m+6), A+8*l+7, A+8*(l+1)+7);
    for (int j = m+1; j >= m; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+8*l+2, A+8*(l+1)+2);
    for (int j = m+5; j >= m+4; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+8*l+6, A+8*(l+1)+6);
    for (int j = m+3; j >= m; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_AVX(RP->s(l, j), RP->c(l, j), A+8*l+4, A+8*(l+1)+4);
    for (int j = m-1; j >= 0; j--)
        for (int l = n-2-j; l >= 0; l--)
            apply_givens_AVX512(RP->s(l, j), RP->c(l, j), A+8*l, A+8*(l+1));
}

// Convert four vectors of triangular harmonics of order 0 to m, m+1, m+2, m+3, m+4, m+5, m+6, m+7.

void kernel_tri_lo2hi_AVX512(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = 0; j < m; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_AVX512(RP->s(l, j), RP->c(l, j), A+8*l, A+8*(l+1));
    for (int j = m; j <= m+3; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_AVX(RP->s(l, j), RP->c(l, j), A+8*l+4, A+8*(l+1)+4);
    for (int j = m+4; j <= m+5; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+8*l+6, A+8*(l+1)+6);
    for (int j = m; j <= m+1; j++)
        for (int l = 0; l <= n-2-j; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+8*l+2, A+8*(l+1)+2);
    for (int l = 0; l <= n-8-m; l++)
        apply_givens_t(RP->s(l, m+6), RP->c(l, m+6), A+8*l+7, A+8*(l+1)+7);
    for (int l = 0; l <= n-6-m; l++)
        apply_givens_t(RP->s(l, m+4), RP->c(l, m+4), A+8*l+5, A+8*(l+1)+5);
    for (int l = 0; l <= n-4-m; l++)
        apply_givens_t(RP->s(l, m+2), RP->c(l, m+2), A+8*l+3, A+8*(l+1)+3);
    for (int l = 0; l <= n-2-m; l++)
        apply_givens_t(RP->s(l, m), RP->c(l, m), A+8*l+1, A+8*(l+1)+1);
}


static inline void apply_givens(const double S, const double C, double * X, double * Y) {
    double x = C*X[0] + S*Y[0];
    double y = C*Y[0] - S*X[0];

    X[0] = x;
    Y[0] = y;
}

static inline void apply_givens_t(const double S, const double C, double * X, double * Y) {
    double x = C*X[0] - S*Y[0];
    double y = C*Y[0] + S*X[0];

    X[0] = x;
    Y[0] = y;
}

#if __SSE2__
    static inline void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        double2 s = vall2(S);
        double2 c = vall2(C);

        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, c*x + s*y);
        vstore2(Y, c*y - s*x);
    }

    static inline void apply_givens_t_SSE(const double S, const double C, double * X, double * Y) {
        double2 s = vall2(S);
        double2 c = vall2(C);

        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, c*x - s*y);
        vstore2(Y, c*y + s*x);
    }
#endif

#if __AVX__
    static inline void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        double4 s = vall4(S);
        double4 c = vall4(C);

        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, c*x + s*y);
        vstore4(Y, c*y - s*x);
    }

    static inline void apply_givens_t_AVX(const double S, const double C, double * X, double * Y) {
        double4 s = vall4(S);
        double4 c = vall4(C);

        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, c*x - s*y);
        vstore4(Y, c*y + s*x);
    }
#endif

#if __AVX512F__
    static inline void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        double8 s = vall8(S);
        double8 c = vall8(C);

        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, c*x + s*y);
        vstore8(Y, c*y - s*x);
    }

    static inline void apply_givens_t_AVX512(const double S, const double C, double * X, double * Y) {
        double8 s = vall8(S);
        double8 c = vall8(C);

        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, c*x - s*y);
        vstore8(Y, c*y + s*x);
    }
#endif
