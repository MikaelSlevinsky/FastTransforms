// Computational kernels for the harmonic polynomial connection problem.

#include "fasttransforms.h"
#include "ftinternal.h"

void ft_destroy_rotation_plan(ft_rotation_plan * RP) {
    free(RP->s);
    free(RP->c);
    free(RP);
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

#ifdef __SSE2__
    static inline void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        double2 x = vloadu2(X);
        double2 y = vloadu2(Y);

        vstoreu2(X, C*x + S*y);
        vstoreu2(Y, C*y - S*x);
    }

    static inline void apply_givens_t_SSE(const double S, const double C, double * X, double * Y) {
        double2 x = vloadu2(X);
        double2 y = vloadu2(Y);

        vstoreu2(X, C*x - S*y);
        vstoreu2(Y, C*y + S*x);
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


#ifdef __AVX__
    static inline void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vloadu4(X);
        double4 y = vloadu4(Y);

        vstoreu4(X, C*x + S*y);
        vstoreu4(Y, C*y - S*x);
    }

    static inline void apply_givens_t_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vloadu4(X);
        double4 y = vloadu4(Y);

        vstoreu4(X, C*x - S*y);
        vstoreu4(Y, C*y + S*x);
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

#ifdef __AVX512F__
    static inline void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);

        vstoreu8(X, C*x + S*y);
        vstoreu8(Y, C*y - S*x);
    }

    static inline void apply_givens_t_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);

        vstoreu8(X, C*x - S*y);
        vstoreu8(Y, C*y + S*x);
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

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

ft_rotation_plan * ft_plan_rotsphere(const int n) {
    double * s = malloc(n*(n+1)/2 * sizeof(double));
    double * c = malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2)*(2*l+2*m+5);
            den = (l+2*m+3)*(l+2*m+4);
            s(l, m) = sqrt(nums/den);
            c(l, m) = sqrt(numc/den);
        }
    ft_rotation_plan * RP = malloc(sizeof(ft_rotation_plan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

void ft_kernel_sph_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_sph_hi2lo_default(RP, m1, m2, A, S);
}

void ft_kernel_sph_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_sph_lo2hi_default(RP, m1, m2, A, S);
}

ft_rotation_plan * ft_plan_rottriangle(const int n, const double alpha, const double beta, const double gamma) {
    double * s = malloc(n*(n+1)/2 * sizeof(double));
    double * c = malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+alpha+1);
            numc = (2*m+beta+gamma+2)*(2*l+2*m+alpha+beta+gamma+4);
            den = (l+2*m+beta+gamma+3)*(l+2*m+alpha+beta+gamma+3);
            s(l, m) = sqrt(nums/den);
            c(l, m) = sqrt(numc/den);
        }
    ft_rotation_plan * RP = malloc(sizeof(ft_rotation_plan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

void ft_kernel_tri_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_tri_hi2lo_default(RP, m1, m2, A, S);
}

void ft_kernel_tri_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_tri_lo2hi_default(RP, m1, m2, A, S);
}

#undef s
#undef c

#define s(l,m) s[l+(m)*n-(m)/2*((m)+1)/2]
#define c(l,m) c[l+(m)*n-(m)/2*((m)+1)/2]

ft_rotation_plan * ft_plan_rotdisk(const int n) {
    double * s = malloc(n*n * sizeof(double));
    double * c = malloc(n*n * sizeof(double));
    double numc, den;
    for (int m = 0; m < 2*n-1; m++)
        for (int l = 0; l < n-(m+1)/2; l++) {
            numc = (m+1)*(2*l+m+3);
            den = (l+m+2)*(l+m+2);
            s(l, m) = -((double) (l+1))/((double) (l+m+2));
            c(l, m) = sqrt(numc/den);
        }
    ft_rotation_plan * RP = malloc(sizeof(ft_rotation_plan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

void ft_kernel_disk_hi2lo(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens(RP->s(l, j), RP->c(l, j), A+l, A+l+1);
}

void ft_kernel_disk_lo2hi(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t(RP->s(l, j), RP->c(l, j), A+l, A+l+1);
}

void ft_kernel_disk_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+1));
}

void ft_kernel_disk_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t_SSE(RP->s(l, j), RP->c(l, j), A+2*l, A+2*(l+1));
}

void ft_kernel_disk_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-2-(m+1)/2; l >= 0; l--)
        apply_givens_SSE(RP->s(l, m), RP->c(l, m), A+4*l+2, A+4*(l+1)+2);
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens_AVX(RP->s(l, j), RP->c(l, j), A+4*l, A+4*(l+1));
}

void ft_kernel_disk_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t_AVX(RP->s(l, j), RP->c(l, j), A+4*l, A+4*(l+1));
    for (int l = 0; l <= n-2-(m+1)/2; l++)
        apply_givens_t_SSE(RP->s(l, m), RP->c(l, m), A+4*l+2, A+4*(l+1)+2);
}

void ft_kernel_disk_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int l = n-2-(m+1)/2; l >= 0; l--)
        apply_givens_SSE(RP->s(l, m), RP->c(l, m), A+8*l+2, A+8*(l+1)+2);
    for (int l = n-4-(m+1)/2; l >= 0; l--)
        apply_givens_SSE(RP->s(l, m+4), RP->c(l, m+4), A+8*l+6, A+8*(l+1)+6);
    for (int j = m+2; j >= m; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens_AVX(RP->s(l, j), RP->c(l, j), A+8*l+4, A+8*(l+1)+4);
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-2-(j+1)/2; l >= 0; l--)
            apply_givens_AVX512(RP->s(l, j), RP->c(l, j), A+8*l, A+8*(l+1));
}

void ft_kernel_disk_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t_AVX512(RP->s(l, j), RP->c(l, j), A+8*l, A+8*(l+1));
    for (int j = m; j <= m+2; j += 2)
        for (int l = 0; l <= n-2-(j+1)/2; l++)
            apply_givens_t_AVX(RP->s(l, j), RP->c(l, j), A+8*l+4, A+8*(l+1)+4);
    for (int l = 0; l <= n-4-(m+1)/2; l++)
        apply_givens_t_SSE(RP->s(l, m+4), RP->c(l, m+4), A+8*l+6, A+8*(l+1)+6);
    for (int l = 0; l <= n-2-(m+1)/2; l++)
        apply_givens_t_SSE(RP->s(l, m), RP->c(l, m), A+8*l+2, A+8*(l+1)+2);
}

#undef s
#undef c

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

void ft_kernel_tet_hi2lo(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    double s, c;
    for (int j = m-1; j >= 0; j--) {
        for (int l = L-2-j; l >= 0; l--) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n; k++)
                apply_givens(s, c, A+k+n*l, A+k+n*(l+1));
        }
    }
}

void ft_kernel_tet_lo2hi(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    double s, c;
    for (int j = 0; j < m; j++) {
        for (int l = 0; l <= L-2-j; l++) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n; k++)
                apply_givens_t(s, c, A+k+n*l, A+k+n*(l+1));
        }
    }
}

void ft_kernel_tet_hi2lo_SSE(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = m-1; j >= 0; j--) {
        for (int l = L-2-j; l >= 0; l--) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%2; k += 2)
                apply_givens_SSE(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void ft_kernel_tet_lo2hi_SSE(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = 0; j < m; j++) {
        for (int l = 0; l <= L-2-j; l++) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%2; k += 2)
                apply_givens_t_SSE(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens_t(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void ft_kernel_tet_hi2lo_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = m-1; j >= 0; j--) {
        for (int l = L-2-j; l >= 0; l--) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%4; k += 4)
                apply_givens_AVX(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%4; k < n-n%2; k += 2)
                apply_givens_SSE(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void ft_kernel_tet_lo2hi_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = 0; j < m; j++) {
        for (int l = 0; l <= L-2-j; l++) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%4; k += 4)
                apply_givens_t_AVX(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%4; k < n-n%2; k += 2)
                apply_givens_t_SSE(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens_t(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void ft_kernel_tet_hi2lo_AVX512(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = m-1; j >= 0; j--) {
        for (int l = L-2-j; l >= 0; l--) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%8; k += 8)
                apply_givens_AVX512(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%8; k < n-n%4; k += 4)
                apply_givens_AVX(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%4; k < n-n%2; k += 2)
                apply_givens_SSE(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void ft_kernel_tet_lo2hi_AVX512(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = 0; j < m; j++) {
        for (int l = 0; l <= L-2-j; l++) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%8; k += 8)
                apply_givens_t_AVX512(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%8; k < n-n%4; k += 4)
                apply_givens_t_AVX(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%4; k < n-n%2; k += 2)
                apply_givens_t_SSE(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens_t(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}


void ft_destroy_spin_rotation_plan(ft_spin_rotation_plan * SRP) {
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

ft_spin_rotation_plan * ft_plan_rotspinsphere(const int n, const int s) {
    int as = abs(s);
    double nums, numc, den;

    // The tail
    double * s1 = calloc(2*n*n, sizeof(double));
    double * c1 = calloc(2*n*n, sizeof(double));

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
    double * s2 = calloc(n*(as+1)*(as+2)/2, sizeof(double));
    double * c2 = calloc(n*(as+1)*(as+2)/2, sizeof(double));

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
    double * s3 = calloc(n*as, sizeof(double));
    double * c3 = calloc(n*as, sizeof(double));

    for (int m = 0; m < as; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2)*(2*l+2*m+5);
            den = (l+2*m+3)*(l+2*m+4);
            s3(l, m) = sqrt(nums/den);
            c3(l, m) = sqrt(numc/den);
        }

    ft_spin_rotation_plan * SRP = malloc(sizeof(ft_spin_rotation_plan));
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

void ft_kernel_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, const int m, double * A) {
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

void ft_kernel_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, const int m, double * A) {
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

void ft_kernel_spinsph_hi2lo_SSE(const ft_spin_rotation_plan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = as+am-2;
    int flick = j%2;

    while (j >= 2*as) {
        for (int l = n-3+as-j; l >= 0; l--)
            apply_givens_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+2*l, A+2*(l+1));
        for (int l = n-2+as-j; l >= 0; l--)
            apply_givens_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+2*l, A+2*(l+1));
        j -= 2;
    }
    while (j >= MAX(0, as-am)) {
        for (int l = n-2-MAX(0, as-am)/2-flick-j/2; l >= 0; l--)
            apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+2*l, A+2*(l+1));
        j -= 2;
    }
    while (j >= 0) {
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+2*l, A+2*(l+2));
        j -= 2;
    }
}

void ft_kernel_spinsph_lo2hi_SSE(const ft_spin_rotation_plan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = (as+am)%2;
    int flick = j;

    while (j < MAX(0, as-am)) {
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+2*l, A+2*(l+2));
        j += 2;
    }
    while (j < MIN(2*as, as+am)) {
        for (int l = 0; l <= n-2-MAX(0, as-am)/2-flick-j/2; l++)
            apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+2*l, A+2*(l+1));
        j += 2;
    }
    while (j < as + am) {
        for (int l = 0; l <= n-2+as-j; l++)
            apply_givens_t_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+2*l, A+2*(l+1));
        for (int l = 0; l <= n-3+as-j; l++)
            apply_givens_t_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+2*l, A+2*(l+1));
        j += 2;
    }
}

void ft_kernel_spinsph_hi2lo_AVX(const ft_spin_rotation_plan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = as+am;
    int flick = j%2;

    if (am <= (as - 1)) {
        while (j >= MAX(0, as-am-2)) {
            for (int l = n-2-MAX(0, as-am-2)/2-flick-j/2; l >= 0; l--)
                apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am-2)), SRP->c2(l, j, MAX(0, as-am-2)), A+4*l+2, A+4*(l+1)+2);
            j -= 2;
        }
        while (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+4*l+2, A+4*(l+2)+2);
            j -= 2;
        }

        j = as+am-2;

        while (j >= MAX(0, as-am)) {
            for (int l = n-2-MAX(0, as-am)/2-flick-j/2; l >= 0; l--)
                apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+4*l, A+4*(l+1));
            j -= 2;
        }
        while (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+4*l, A+4*(l+2));
            j -= 2;
        }
    } else {
        if (j >= 2*as) {
            for (int l = n-3+as-j; l >= 0; l--)
                apply_givens_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+4*l+2, A+4*(l+1)+2);
            for (int l = n-2+as-j; l >= 0; l--)
                apply_givens_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+4*l+2, A+4*(l+1)+2);
            j -= 2;
        } else if (j >= MAX(0, as-am-2)) {
            for (int l = n-2-MAX(0, as-am-2)/2-flick-j/2; l >= 0; l--)
                apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am-2)), SRP->c2(l, j, MAX(0, as-am-2)), A+4*l+2, A+4*(l+1)+2);
            j -= 2;
        } else if (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+4*l+2, A+4*(l+2)+2);
            j -= 2;
        }

        while (j >= 2*as) {
            for (int l = n-3+as-j; l >= 0; l--)
                apply_givens_AVX(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+4*l, A+4*(l+1));
            for (int l = n-2+as-j; l >= 0; l--)
                apply_givens_AVX(SRP->s1(l, j-as), SRP->c1(l, j-as), A+4*l, A+4*(l+1));
            j -= 2;
        }
        while (j >= MAX(0, as-am)) {
            for (int l = n-2-MAX(0, as-am)/2-flick-j/2; l >= 0; l--)
                apply_givens_AVX(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+4*l, A+4*(l+1));
            j -= 2;
        }
        while (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_AVX(SRP->s3(l, j), SRP->c3(l, j), A+4*l, A+4*(l+2));
            j -= 2;
        }
    }
}

void ft_kernel_spinsph_lo2hi_AVX(const ft_spin_rotation_plan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = (as+am)%2;
    int flick = j;

   if (am > (as - 1)) {
        while (j < MAX(0, as-am)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_AVX(SRP->s3(l, j), SRP->c3(l, j), A+4*l, A+4*(l+2));
            j += 2;
        }
        while (j < MIN(2*as, as+am)) {
            for (int l = 0; l <= n-2-MAX(0, as-am)/2-flick-j/2; l++)
                apply_givens_t_AVX(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+4*l, A+4*(l+1));
            j += 2;
        }
        while (j < as + am) {
            for (int l = 0; l <= n-2+as-j; l++)
                apply_givens_t_AVX(SRP->s1(l, j-as), SRP->c1(l, j-as), A+4*l, A+4*(l+1));
            for (int l = 0; l <= n-3+as-j; l++)
                apply_givens_t_AVX(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+4*l, A+4*(l+1));
            j += 2;
        }

        if (j < MAX(0, as-am-2)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+4*l+2, A+4*(l+2)+2);
            j += 2;
        } else if (j < MIN(2*as, as+am+2)) {
            for (int l = 0; l <= n-2-MAX(0, as-am)/2-flick-j/2; l++)
                apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am-2)), SRP->c2(l, j, MAX(0, as-am-2)), A+4*l+2, A+4*(l+1)+2);
            j += 2;
        } else if (j < as + am + 2) {
            for (int l = 0; l <= n-2+as-j; l++)
                apply_givens_t_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+4*l+2, A+4*(l+1)+2);
            for (int l = 0; l <= n-3+as-j; l++)
                apply_givens_t_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+4*l+2, A+4*(l+1)+2);
            j += 2;
        }
   } else {
        while (j < MAX(0, as-am)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+4*l, A+4*(l+2));
            j += 2;
        }
        while (j < MIN(2*as, as+am)) {
            for (int l = 0; l <= n-2-MAX(0, as-am)/2-flick-j/2; l++)
                apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+4*l, A+4*(l+1));
            j += 2;
        }

        j = (as+am)%2;

        while (j < MAX(0, as-am-2)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+4*l+2, A+4*(l+2)+2);
            j += 2;
        }
        while (j < MIN(2*as, as+am+2)) {
            for (int l = 0; l <= n-2-MAX(0, as-am-2)/2-flick-j/2; l++)
                apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am-2)), SRP->c2(l, j, MAX(0, as-am-2)), A+4*l+2, A+4*(l+1)+2);
            j += 2;
        }
    }
}

void ft_kernel_spinsph_hi2lo_AVX512(const ft_spin_rotation_plan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = as+am+4;
    int flick = j%2;

    if (am <= (as - 1)) {
        for (int i = 0; i <= 6; i += 2) {
            j = as + am + (i-2);
            while (j >= 2*as) {
                for (int l = n-3+as-j; l >= 0; l--)
                    apply_givens_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+i, A+8*(l+1)+i);
                for (int l = n-2+as-j; l >= 0; l--)
                    apply_givens_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+i, A+8*(l+1)+i);
                j -= 2;
            }
            while (j >= MAX(0, as-am-i)) {
                for (int l = n-2-MAX(0, as-am-i)/2-flick-j/2; l >= 0; l--)
                    apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am-i)), SRP->c2(l, j, MAX(0, as-am-i)), A+8*l+i, A+8*(l+1)+i);
                j -= 2;
            }
            while (j >= 0) {
                for (int l = n-3-j; l >= 0; l--)
                    apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+8*l+i, A+8*(l+2)+i);
                j -= 2;
            }
        }
    }
    else {
        if (j >= 2*as) {
            for (int l = n-3+as-j; l >= 0; l--)
                apply_givens_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+6, A+8*(l+1)+6);
            for (int l = n-2+as-j; l >= 0; l--)
                apply_givens_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+6, A+8*(l+1)+6);
            j -= 2;
        } else if (j >= MAX(0, as-am-6)) {
            for (int l = n-2-MAX(0, as-am-6)/2-flick-j/2; l >= 0; l--)
                apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am-6)), SRP->c2(l, j, MAX(0, as-am-6)), A+8*l+6, A+8*(l+1)+6);
            j -= 2;
        } else if (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+8*l+6, A+8*(l+2)+6);
            j -= 2;
        }

        for (int i = 2; i > 0; i--) {
            if (j >= 2*as) {
                for (int l = n-3+as-j; l >= 0; l--)
                    apply_givens_AVX(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+4, A+8*(l+1)+4);
                for (int l = n-2+as-j; l >= 0; l--)
                    apply_givens_AVX(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+4, A+8*(l+1)+4);
                j -= 2;
            } else if (j >= MAX(0, as-am-i*2)) {
                for (int l = n-2-MAX(0, as-am-4)/2-flick-j/2; l >= 0; l--)
                    apply_givens_AVX(SRP->s2(l, j, MAX(0, as-am-4)), SRP->c2(l, j, MAX(0, as-am-4)), A+8*l+4, A+8*(l+1)+4);
                j -= 2;
            } else if (j >= 0) {
                for (int l = n-3-j; l >= 0; l--)
                    apply_givens_AVX(SRP->s3(l, j), SRP->c3(l, j), A+8*l+4, A+8*(l+2)+4);
                j -= 2;
            }
        }

        j = as + am;

        if (j >= 2*as) {
            for (int l = n-3+as-j; l >= 0; l--)
                apply_givens_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+2, A+8*(l+1)+2);
            for (int l = n-2+as-j; l >= 0; l--)
                apply_givens_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+2, A+8*(l+1)+2);
            j -= 2;
        } else if (j >= MAX(0, as-am-2)) {
            for (int l = n-2-MAX(0, as-am-2)/2-flick-j/2; l >= 0; l--)
                apply_givens_SSE(SRP->s2(l, j, MAX(0, as-am-2)), SRP->c2(l, j, MAX(0, as-am-2)), A+8*l+2, A+8*(l+1)+2);
            j -= 2;
        } else if (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_SSE(SRP->s3(l, j), SRP->c3(l, j), A+8*l+2, A+8*(l+2)+2);
            j -= 2;
        }

        while (j >= 2*as) {
            for (int l = n-3+as-j; l >= 0; l--)
                apply_givens_AVX512(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l, A+8*(l+1));
            for (int l = n-2+as-j; l >= 0; l--)
                apply_givens_AVX512(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l, A+8*(l+1));
            j -= 2;
        }
        while (j >= MAX(0, as-am)) {
            for (int l = n-2-MAX(0, as-am)/2-flick-j/2; l >= 0; l--)
                apply_givens_AVX512(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+8*l, A+8*(l+1));
            j -= 2;
        }
        while (j >= 0) {
            for (int l = n-3-j; l >= 0; l--)
                apply_givens_AVX512(SRP->s3(l, j), SRP->c3(l, j), A+8*l, A+8*(l+2));
            j -= 2;
        }
    }
}

void ft_kernel_spinsph_lo2hi_AVX512(const ft_spin_rotation_plan * SRP, const int m, double * A) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    int j = (as+am)%2;
    int flick = j;

   if (am > (as - 1)) {
        while (j < MAX(0, as-am)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_AVX512(SRP->s3(l, j), SRP->c3(l, j), A+8*l, A+8*(l+2));
            j += 2;
        }
        while (j < MIN(2*as, as+am)) {
            for (int l = 0; l <= n-2-MAX(0, as-am)/2-flick-j/2; l++)
                apply_givens_t_AVX512(SRP->s2(l, j, MAX(0, as-am)), SRP->c2(l, j, MAX(0, as-am)), A+8*l, A+8*(l+1));
            j += 2;
        }
        while (j < as + am) {
            for (int l = 0; l <= n-2+as-j; l++)
                apply_givens_t_AVX512(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l, A+8*(l+1));
            for (int l = 0; l <= n-3+as-j; l++)
                apply_givens_t_AVX512(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l, A+8*(l+1));
            j += 2;
        }

        if (j < MAX(0, as-am-2)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+8*l+2, A+8*(l+2)+2);
        } else if (j < MIN(2*as, as+am+2)) {
            for (int l = 0; l <= n-2-MAX(0, as-am-2)/2-flick-j/2; l++)
                apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am-2)), SRP->c2(l, j, MAX(0, as-am-2)), A+8*l+2, A+8*(l+1)+2);
        } else if (j < as + am + 2) {
            for (int l = 0; l <= n-2+as-j; l++)
                apply_givens_t_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+2, A+8*(l+1)+2);
            for (int l = 0; l <= n-3+as-j; l++)
                apply_givens_t_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+2, A+8*(l+1)+2);
        }

        for (int i = 2; i <= 4; i+=2) {
            if (j < MAX(0, as-am-i)) {
                for (int l = 0; l <= n-3-j; l++)
                    apply_givens_t_AVX(SRP->s3(l, j), SRP->c3(l, j), A+8*l+4, A+8*(l+2)+4);
                j += 2;
            } else if (j < MIN(2*as, as+am+i)) {
                for (int l = 0; l <= n-2-MAX(0, as-am-i)/2-flick-j/2; l++)
                    apply_givens_t_AVX(SRP->s2(l, j, MAX(0, as-am-i)), SRP->c2(l, j, MAX(0, as-am-i)), A+8*l+4, A+8*(l+1)+4);
                j += 2;
            } else if (j < as + am + i) {
                for (int l = 0; l <= n-2+as-j; l++)
                    apply_givens_t_AVX(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+4, A+8*(l+1)+4);
                for (int l = 0; l <= n-3+as-j; l++)
                    apply_givens_t_AVX(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+4, A+8*(l+1)+4);
                j += 2;
            }
        }

        if (j < MAX(0, as-am-6)) {
            for (int l = 0; l <= n-3-j; l++)
                apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+8*l+6, A+8*(l+2)+6);
        } else if (j < MIN(2*as, as+am+6)) {
            for (int l = 0; l <= n-2-MAX(0, as-am-6)/2-flick-j/2; l++)
                apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am-6)), SRP->c2(l, j, MAX(0, as-am-6)), A+8*l+6, A+8*(l+1)+6);
        } else if (j < as + am + 6) {
            for (int l = 0; l <= n-2+as-j; l++)
                apply_givens_t_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+6, A+8*(l+1)+6);
            for (int l = 0; l <= n-3+as-j; l++)
                apply_givens_t_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+6, A+8*(l+1)+6);
        }
   } else {
        for (int i = 0; i <= 6; i += 2) {
            j = (as+am)%2;
            while (j < MAX(0, as-am-i)) {
                for (int l = 0; l <= n-3-j; l++)
                    apply_givens_t_SSE(SRP->s3(l, j), SRP->c3(l, j), A+8*l+i, A+8*(l+2)+i);
                j += 2;
            }
            while (j < MIN(2*as, as+am+i)) {
                for (int l = 0; l <= n-2-MAX(0, as-am-i)/2-flick-j/2; l++)
                    apply_givens_t_SSE(SRP->s2(l, j, MAX(0, as-am-i)), SRP->c2(l, j, MAX(0, as-am-i)), A+8*l+i, A+8*(l+1)+i);
                j += 2;
            }
            while (j < as + am + i) {
                for (int l = 0; l <= n-2+as-j; l++)
                    apply_givens_t_SSE(SRP->s1(l, j-as), SRP->c1(l, j-as), A+8*l+i, A+8*(l+1)+i);
                for (int l = 0; l <= n-3+as-j; l++)
                    apply_givens_t_SSE(SRP->s1(l+n, j-as), SRP->c1(l+n, j-as), A+8*l+i, A+8*(l+1)+i);
                j += 2;
            }
        }
    }
}
