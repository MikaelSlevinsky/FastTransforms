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

#define sd(l,m) s[l+(m)*n-(m)/2*((m)+1)/2]
#define cd(l,m) c[l+(m)*n-(m)/2*((m)+1)/2]

ft_rotation_plan * ft_plan_rotdisk(const int n) {
    double * s = malloc(n*n * sizeof(double));
    double * c = malloc(n*n * sizeof(double));
    double numc, den;
    for (int m = 0; m < 2*n-1; m++)
        for (int l = 0; l < n-(m+1)/2; l++) {
            numc = (m+1)*(2*l+m+3);
            den = (l+m+2)*(l+m+2);
            sd(l, m) = -((double) (l+1))/((double) (l+m+2));
            cd(l, m) = sqrt(numc/den);
        }
    ft_rotation_plan * RP = malloc(sizeof(ft_rotation_plan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

void ft_kernel_disk_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_disk_hi2lo_default(RP, m1, m2, A, S);
}

void ft_kernel_disk_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_disk_lo2hi_default(RP, m1, m2, A, S);
}


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
    free(SRP);
}

#define s1(l,m) s1[l+(m)*n]
#define c1(l,m) c1[l+(m)*n]

#define s2(l,j,m) s2[l+(j)*n+(m)*as*n]
#define c2(l,j,m) c2[l+(j)*n+(m)*as*n]

ft_spin_rotation_plan * ft_plan_rotspinsphere(const int n, const int s) {
    int as = abs(s);
    double nums, numc, den;
    double * s1 = calloc(n*n, sizeof(double));
    double * c1 = calloc(n*n, sizeof(double));
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2)*(2*l+2*m+5);
            den = (l+2*m+3)*(l+2*m+4);
            s1(l, m) = sqrt(nums/den);
            c1(l, m) = sqrt(numc/den);
        }
    double * s2 = calloc(as*n*n, sizeof(double));
    double * c2 = calloc(as*n*n, sizeof(double));
    for (int m = 0; m < n; m++)
        for (int j = 0; j < as; j++)
            for (int l = 0; l < n-m-j; l++) {
                nums = (l+1)*(l+m+1);
                numc = (2*j+m+1)*(2*l+2*j+2*m+3);
                den = (l+2*j+m+2)*(l+2*j+2*m+2);
                s2(l, j, m) = sqrt(nums/den);
                c2(l, j, m) = sqrt(numc/den);
        }
    ft_spin_rotation_plan * SRP = malloc(sizeof(ft_spin_rotation_plan));
    SRP->s1 = s1;
    SRP->c1 = c1;
    SRP->s2 = s2;
    SRP->c2 = c2;
    SRP->n = n;
    SRP->s = s;
    return SRP;
}

void ft_kernel_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
    kernel_spinsph_hi2lo_SSE2(SRP, m, A, S);
}

void ft_kernel_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
    kernel_spinsph_lo2hi_SSE2(SRP, m, A, S);
}
