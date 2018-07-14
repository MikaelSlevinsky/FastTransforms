// Computational routines for the harmonic polynomial connection problem.

#include "fasttransforms.h"

void freeRotationPlan(RotationPlan * RP) {
    free(RP->s);
    free(RP->c);
    free(RP);
}

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

void kernel_sph_hi2lo_AVX512(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-3-m; l >= 0; l--)
        apply_givens_SSE(RP->s(l, m), RP->c(l, m), A+8*l+2, A+8*(l+2)+2);
    for (int l = n-7-m; l >= 0; l--)
        apply_givens_SSE(RP->s(l, m+4), RP->c(l, m+4), A+8*l+6, A+8*(l+2)+6);
    for(int j = m+2; j >= m; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_AVX(RP->s(l, j), RP->c(l, j), A+8*l+4, A+8*(l+2)+4);
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_AVX512(RP->s(l, j), RP->c(l, j), A+8*l, A+8*(l+2));
}

void kernel_sph_lo2hi_AVX512(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_AVX512(RP->s(l, j), RP->c(l, j), A+8*l, A+8*(l+2));
    for(int j = m; j <= m+2; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_AVX(RP->s(l, j), RP->c(l, j), A+8*l+4, A+8*(l+2)+4);
    for (int l = 0; l <= n-7-m; l++)
        apply_givens_t_SSE(RP->s(l, m+4), RP->c(l, m+4), A+8*l+6, A+8*(l+2)+6);
    for (int l = 0; l <= n-3-m; l++)
        apply_givens_t_SSE(RP->s(l, m), RP->c(l, m), A+8*l+2, A+8*(l+2)+2);
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

#undef s
#undef c

#define s(l,m) s[l+(m)*n-(m)/2*((m)+1)/2]
#define c(l,m) c[l+(m)*n-(m)/2*((m)+1)/2]

RotationPlan * plan_rotdisk(const int n) {
    double * s = (double *) malloc(n*n * sizeof(double));
    double * c = (double *) malloc(n*n * sizeof(double));
    double numc, den;
    for (int m = 0; m < 2*n-1; m++)
        for (int l = 0; l < n-(m+1)/2; l++) {
            numc = (m+1)*(2*l+m+3);
            den = (l+m+2)*(l+m+2);
            s(l, m) = -((double) (l+1))/((double) (l+m+2));
            c(l, m) = sqrt(numc/den);
        }
    RotationPlan * RP = malloc(sizeof(RotationPlan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

// Convert a single vector of disk harmonics of order m to 0/1.

void kernel_disk_hi2lo(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens(RP->s(l, j), RP->c(l, j), A+l, A+l+1);
}

// Convert a single vector of disk harmonics of order 0/1 to m.

void kernel_disk_lo2hi(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t(RP->s(l, j), RP->c(l, j), A+l, A+l+1);
}

// Convert a pair of vectors of disk harmonics of order m to 0/1.

void kernel_disk_hi2lo_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+1));
}

// Convert a pair of vectors of disk harmonics of order 0/1 to m.

void kernel_disk_lo2hi_SSE(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+1));
}


void freeSpinRotationPlan(SpinRotationPlan * SRP) {
    free(SRP->s1);
    free(SRP->c1);
    free(SRP->s2);
    free(SRP->c2);
    free(SRP->s3);
    free(SRP->c3);
    free(SRP);
}

#undef s
#undef c

#define s1(l,m) s1[l+(m)*2*n]
#define c1(l,m) c1[l+(m)*2*n]

#define s2(l,k,m) s2[l+((k-(m))/2+(as+1)*(as+2)/2-(as+1-(m))*(as+2-(m))/2)*n]
#define c2(l,k,m) c2[l+((k-(m))/2+(as+1)*(as+2)/2-(as+1-(m))*(as+2-(m))/2)*n]

#define s3(l,m) s3[l+(m)*n]
#define c3(l,m) c3[l+(m)*n]

SpinRotationPlan * plan_rotspinsphere(const int n, const int s) {
    int as = abs(s);
    double nums, numc, den;

    // The tail
    double * s1 = (double *) calloc(2*n*n, sizeof(double));
    double * c1 = (double *) calloc(2*n*n, sizeof(double));

    for (int m = as; m < n+as; m++)
        for (int l = 0; l < n; l++) {
            // Down
            nums = (l+1)*(l+m+as+1);
            numc = (m-as+1)*(2*l+2*m+3);
            den = (l+m-as+2)*(l+2*m+2);
            s1(l, m-as) = -sqrt(nums/den);
            c1(l, m-as) = sqrt(numc/den);
            // Left
            nums = (l+1)*(l+m-as+3);
            numc = (m+as+1)*(2*l+2*m+5);
            den = (l+m+as+2)*(l+2*m+4);
            s1(l+n, m-as) = sqrt(nums/den);
            c1(l+n, m-as) = sqrt(numc/den);
        }

    // The O(s^2) triangle
    double * s2 = (double *) calloc(n*(as+1)*(as+2)/2, sizeof(double));
    double * c2 = (double *) calloc(n*(as+1)*(as+2)/2, sizeof(double));

    for (int m = 0; m < as+1; m++)
        for (int k = m; k < 2*as+2-m; k += 2)
            for (int l = 0; l < n-(k-m)/2; l++) {
                nums = (l+1)*(l+m+1);
                numc = (k+1)*(2*l+k+m+3);
                den = (l+k+2)*(l+k+m+2);
                s2(l, k, m) = sqrt(nums/den);
                c2(l, k, m) = sqrt(numc/den);
            }

    // The main diagonal
    double * s3 = (double *) calloc(n*as, sizeof(double));
    double * c3 = (double *) calloc(n*as, sizeof(double));

    for (int m = 0; m < as; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2)*(2*l+2*m+5);
            den = (l+2*m+3)*(l+2*m+4);
            s3(l, m) = sqrt(nums/den);
            c3(l, m) = sqrt(numc/den);
        }

    SpinRotationPlan * SRP = malloc(sizeof(SpinRotationPlan));
    SRP->s1 = s1;
    SRP->c1 = c1;
    SRP->s2 = s2;
    SRP->c2 = c2;
    SRP->s3 = s3;
    SRP->c3 = c3;
    SRP->n = n;
    SRP->s = s;
    return SRP;
}

// Convert a single vector of spin-weighted spherical harmonics of order m to 0/1.

void kernel_spinsph_hi2lo(const SpinRotationPlan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = as+am-2;
    int flick = j%2;

    while (j >= 2*as) {
        for (int l = n-3+as-j; l >= 0; l--)
            apply_givens(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+l, A+l+1);
        for (int l = n-2+as-j; l >= 0; l--)
            apply_givens(SRP->s1(l, j-as), SRP->c1(l, j-as), A+l, A+l+1);
        j -= 2;
    }
    while (j >= MAX(0, as-am)) {
        for (int l = n-2-MAX(0, as-am)/2-flick-j/2; l >= 0; l--)
            apply_givens(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+l, A+l+1);
        j -= 2;
    }
    while (j >= 0) {
        for (int l = n-3-j; l >= 0; l--)
            apply_givens(SRP->s3(l, j), SRP->c3(l, j), A+l, A+l+2);
        j -= 2;
    }
}

// Convert a single vector of spin-weighted spherical harmonics of order m to 0/1.

void kernel_spinsph_lo2hi(const SpinRotationPlan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = (as+am)%2;
    int flick = j;

    while (j < MAX(0, as-am)) {
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t(SRP->s3(l, j), SRP->c3(l, j), A+l, A+l+2);
        j += 2;
    }
    while (j < MIN(2*as, as+am)) {
        for (int l = 0; l <= n-2-MAX(0, as-am)/2-flick-j/2; l++)
            apply_givens_t(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+l, A+l+1);
        j += 2;
    }
    while (j < as + am) {
        for (int l = 0; l <= n-2+as-j; l++)
            apply_givens_t(SRP->s1(l, j-as), SRP->c1(l, j-as), A+l, A+l+1);
        for (int l = 0; l <= n-3+as-j; l++)
            apply_givens_t(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+l, A+l+1);
        j += 2;
    }
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
        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, C*x + S*y);
        vstore2(Y, C*y - S*x);
    }

    static inline void apply_givens_t_SSE(const double S, const double C, double * X, double * Y) {
        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, C*x - S*y);
        vstore2(Y, C*y + S*x);
    }
#else
    static inline void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        apply_givens(S, C, X, Y);
        apply_givens(S, C, X+1, Y+1);
    }

    static inline void apply_givens_t_SSE(const double S, const double C, double * X, double * Y) {
        apply_givens_t(S, C, X, Y);
        apply_givens_t(S, C, X+1, Y+1);
    }
#endif


#if __AVX__
    static inline void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, C*x + S*y);
        vstore4(Y, C*y - S*x);
    }

    static inline void apply_givens_t_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, C*x - S*y);
        vstore4(Y, C*y + S*x);
    }
#else
    static inline void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        apply_givens_SSE(S, C, X, Y);
        apply_givens_SSE(S, C, X+2, Y+2);
    }

    static inline void apply_givens_t_AVX(const double S, const double C, double * X, double * Y) {
        apply_givens_t_SSE(S, C, X, Y);
        apply_givens_t_SSE(S, C, X+2, Y+2);
    }
#endif

#if __AVX512F__
    static inline void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, C*x + S*y);
        vstore8(Y, C*y - S*x);
    }

    static inline void apply_givens_t_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, C*x - S*y);
        vstore8(Y, C*y + S*x);
    }
#else
    static inline void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        apply_givens_AVX(S, C, X, Y);
        apply_givens_AVX(S, C, X+4, Y+4);
    }

    static inline void apply_givens_t_AVX512(const double S, const double C, double * X, double * Y) {
        apply_givens_t_AVX(S, C, X, Y);
        apply_givens_t_AVX(S, C, X+4, Y+4);
    }
#endif
