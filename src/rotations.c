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
    static inline void apply_givens_SSE2(const double S, const double C, double * X, double * Y) {
        double2 x = vloadu2(X);
        double2 y = vloadu2(Y);

        vstoreu2(X, C*x + S*y);
        vstoreu2(Y, C*y - S*x);
    }

    static inline void apply_givens_t_SSE2(const double S, const double C, double * X, double * Y) {
        double2 x = vloadu2(X);
        double2 y = vloadu2(Y);

        vstoreu2(X, C*x - S*y);
        vstoreu2(Y, C*y + S*x);
    }
#else
    static inline void apply_givens_SSE2(const double S, const double C, double * X, double * Y) {
        apply_givens(S, C, X, Y);
        apply_givens(S, C, X+1, Y+1);
    }

    static inline void apply_givens_t_SSE2(const double S, const double C, double * X, double * Y) {
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
        apply_givens_SSE2(S, C, X, Y);
        apply_givens_SSE2(S, C, X+2, Y+2);
    }

    static inline void apply_givens_t_AVX(const double S, const double C, double * X, double * Y) {
        apply_givens_t_SSE2(S, C, X, Y);
        apply_givens_t_SSE2(S, C, X+2, Y+2);
    }
#endif

#ifdef __AVX512F__
    static inline void apply_givens_AVX512F(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);

        vstoreu8(X, C*x + S*y);
        vstoreu8(Y, C*y - S*x);
    }

    static inline void apply_givens_t_AVX512F(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);

        vstoreu8(X, C*x - S*y);
        vstoreu8(Y, C*y + S*x);
    }
#else
    static inline void apply_givens_AVX512F(const double S, const double C, double * X, double * Y) {
        apply_givens_AVX(S, C, X, Y);
        apply_givens_AVX(S, C, X+4, Y+4);
    }

    static inline void apply_givens_t_AVX512F(const double S, const double C, double * X, double * Y) {
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

#define sd(l,m) s[l+(m)*n-((m)/2)*(((m)+1)/2)]
#define cd(l,m) c[l+(m)*n-((m)/2)*(((m)+1)/2)]

ft_rotation_plan * ft_plan_rotdisk(const int n, const double alpha, const double beta) {
    double * s = malloc(n*n * sizeof(double));
    double * c = malloc(n*n * sizeof(double));
    double nums, numc, den;
    for (int m = 0; m < 2*n-1; m++)
        for (int l = 0; l < n-(m+1)/2; l++) {
            nums = (l+1)*(l+beta+1);
            numc = (m+alpha+1)*(2*l+m+alpha+beta+3);
            den = (l+m+alpha+2)*(l+m+alpha+beta+2);
            sd(l, m) = -sqrt(nums/den);
            cd(l, m) = sqrt(numc/den);
        }
    ft_rotation_plan * RP = malloc(sizeof(ft_rotation_plan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

// Orthonormal semi-classical Jacobi hierarchy with respect to w(x) = (1-x)^β (1+x)^α (t+x)^(γ+m).
// If t = 1, then the sines and cosines should match (1-x)^β (1+x)^(α+γ+m), which is the rotdisk with (α+γ, β).
ft_rotation_plan * ft_plan_rotannulus(const int n, const double alpha, const double beta, const double gamma, const double rho) {
    double * s = malloc(n*n * sizeof(double));
    double * c = malloc(n*n * sizeof(double));
    double t = 1.0 + 2.0*rho*rho/(1.0-rho*rho);
    if (gamma == 0.0) {
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->nu = 1;
        P->nv = 0;
        ft_banded * XP = ft_create_jacobi_multiplication(1, n+1, n+1, beta, alpha);
        ft_symmetric_tridiagonal * JP = ft_convert_banded_to_symmetric_tridiagonal(XP);
        for (int i = 0; i < n+1; i++)
            JP->a[i] += t;
        for (int m = 0; m < 2*n-1; m += 2) {
            ft_symmetric_tridiagonal_qr * QR = ft_symmetric_tridiagonal_qrfact(JP);
            for (int l = 0; l < n-(m+1)/2; l++) {
                sd(l, m) = QR->s[l];
                cd(l, m) = QR->c[l];
            }
            P->R = QR->R;
            P->n = QR->n;
            ft_symmetric_tridiagonal * JP2 = ft_execute_jacobi_similarity(P, JP);
            ft_destroy_symmetric_tridiagonal(JP);
            JP = JP2;
            ft_destroy_symmetric_tridiagonal_qr(QR);
        }
        XP = ft_create_jacobi_multiplication(1, n+1, n+1, beta, alpha);
        JP = ft_convert_banded_to_symmetric_tridiagonal(XP);
        for (int i = 0; i < n+1; i++)
            JP->a[i] += t;
        XP = ft_create_jacobi_multiplication(1, n+1, n+1, beta, alpha);
        for (int i = 0; i < n+1; i++)
            ft_set_banded_index(XP, t + ft_get_banded_index(XP, i, i), i, i);
        ft_banded_cholfact(XP);
        P->R = ft_convert_banded_to_triangular_banded(XP);
        P->n = XP->n;
        ft_symmetric_tridiagonal * JP1 = ft_execute_jacobi_similarity(P, JP);
        ft_destroy_symmetric_tridiagonal(JP);
        ft_destroy_triangular_banded(P->R);
        JP = JP1;
        for (int m = 1; m < 2*n-1; m += 2) {
            ft_symmetric_tridiagonal_qr * QR = ft_symmetric_tridiagonal_qrfact(JP);
            for (int l = 0; l < n-(m+1)/2; l++) {
                sd(l, m) = QR->s[l];
                cd(l, m) = QR->c[l];
            }
            P->R = QR->R;
            P->n = QR->n;
            ft_symmetric_tridiagonal * JP2 = ft_execute_jacobi_similarity(P, JP);
            ft_destroy_symmetric_tridiagonal(JP);
            JP = JP2;
            ft_destroy_symmetric_tridiagonal_qr(QR);
        }
        free(P);
        ft_destroy_symmetric_tridiagonal(JP);
    }
    else {
        warning("Not implemented.");
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

ft_rotation_plan * ft_plan_rotrectdisk(const int n, const double beta) {
    double * s = malloc(n*(n+1)/2 * sizeof(double));
    double * c = malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2*beta+3)*(2*l+2*m+2*beta+6);
            den = (l+2*m+2*beta+4)*(l+2*m+2*beta+5);
            s(l, m) = sqrt(nums/den);
            c(l, m) = sqrt(numc/den);
        }
    ft_rotation_plan * RP = malloc(sizeof(ft_rotation_plan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

void ft_kernel_rectdisk_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_rectdisk_hi2lo_default(RP, m1, m2, A, S);
}

void ft_kernel_rectdisk_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    kernel_rectdisk_lo2hi_default(RP, m1, m2, A, S);
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

void kernel_tet_hi2lo_SSE2(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = m-1; j >= 0; j--) {
        for (int l = L-2-j; l >= 0; l--) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%2; k += 2)
                apply_givens_SSE2(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void kernel_tet_lo2hi_SSE2(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = 0; j < m; j++) {
        for (int l = 0; l <= L-2-j; l++) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%2; k += 2)
                apply_givens_t_SSE2(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens_t(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void kernel_tet_hi2lo_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A) {
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
                apply_givens_SSE2(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void kernel_tet_lo2hi_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A) {
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
                apply_givens_t_SSE2(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens_t(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void kernel_tet_hi2lo_AVX512F(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = m-1; j >= 0; j--) {
        for (int l = L-2-j; l >= 0; l--) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%8; k += 8)
                apply_givens_AVX512F(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%8; k < n-n%4; k += 4)
                apply_givens_AVX(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%4; k < n-n%2; k += 2)
                apply_givens_SSE2(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%2; k < n; k++)
                apply_givens(s, c, A+k+nb*l, A+k+nb*(l+1));
        }
    }
}

void kernel_tet_lo2hi_AVX512F(const ft_rotation_plan * RP, const int L, const int m, double * A) {
    int n = RP->n;
    int nb = VALIGN(n);
    double s, c;
    for (int j = 0; j < m; j++) {
        for (int l = 0; l <= L-2-j; l++) {
            s = RP->s(l, j);
            c = RP->c(l, j);
            for (int k = 0; k < n-n%8; k += 8)
                apply_givens_t_AVX512F(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%8; k < n-n%4; k += 4)
                apply_givens_t_AVX(s, c, A+k+nb*l, A+k+nb*(l+1));
            for (int k = n-n%4; k < n-n%2; k += 2)
                apply_givens_t_SSE2(s, c, A+k+nb*l, A+k+nb*(l+1));
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
