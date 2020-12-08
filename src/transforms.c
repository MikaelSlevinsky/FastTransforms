// Computational routines for one-dimensional orthogonal polynomial transforms.

#include "fasttransforms.h"
#include "ftinternal.h"

#if defined(FT_QUADMATH)
    #define FLT long double
    #define FLT2 quadruple
    #define X(name) FT_CONCAT(ft_, name, l)
    #define X2(name) FT_CONCAT(ft_, name, q)
    #define Y2(name) FT_CONCAT(, name, q)
    #include "transforms_source.c"
    #undef FLT
    #undef FLT2
    #undef X
    #undef X2
    #undef Y2
#endif

#define FLT double
#define FLT2 long double
#define X(name) FT_CONCAT(ft_, name, )
#define X2(name) FT_CONCAT(ft_, name, l)
#define Y2(name) FT_CONCAT(, name, l)
#include "transforms_source.c"
#undef FLT
#undef FLT2
#undef X
#undef X2
#undef Y2

#define FLT float
#define FLT2 double
#define X(name) FT_CONCAT(ft_, name, f)
#define X2(name) FT_CONCAT(ft_, name, )
#define Y2(name) FT_CONCAT(, name, )
#include "transforms_source.c"
#undef FLT
#undef FLT2
#undef X
#undef X2
#undef Y2

double * plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n) {
    ft_triangular_bandedl * A = ft_create_A_legendre_to_chebyshevl(normcheb, n);
    ft_triangular_bandedl * B = ft_create_B_legendre_to_chebyshevl(normcheb, n);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_legendre_to_chebyshev_diagonal_connection_coefficientl(normleg, normcheb, n, Vl, n+1);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++)
        V[i] = Vl[i];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    return V;
}

double * plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n) {
    ft_triangular_bandedl * A = ft_create_A_chebyshev_to_legendrel(normleg, n);
    ft_triangular_bandedl * B = ft_create_B_chebyshev_to_legendrel(normleg, n);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_chebyshev_to_legendre_diagonal_connection_coefficientl(normcheb, normleg, n, Vl, n+1);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++)
        V[i] = Vl[i];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    return V;
}

double * plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, const double lambda, const double mu) {
    ft_triangular_bandedl * A = ft_create_A_ultraspherical_to_ultrasphericall(norm2, n, lambda, mu);
    ft_triangular_bandedl * B = ft_create_B_ultraspherical_to_ultrasphericall(norm2, n, mu);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_ultraspherical_to_ultraspherical_diagonal_connection_coefficientl(norm1, norm2, n, lambda, mu, Vl, n+1);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++)
        V[i] = Vl[i];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    return V;
}

double * plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const double alpha, const double beta, const double gamma, const double delta) {
    ft_triangular_bandedl * A = ft_create_A_jacobi_to_jacobil(norm2, n, alpha, beta, gamma, delta);
    ft_triangular_bandedl * B = ft_create_B_jacobi_to_jacobil(norm2, n, gamma, delta);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_jacobi_to_jacobi_diagonal_connection_coefficientl(norm1, norm2, n, alpha, beta, gamma, delta, Vl, n+1);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++)
        V[i] = Vl[i];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    return V;
}

double * plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const double alpha, const double beta) {
    ft_triangular_bandedl * A = ft_create_A_laguerre_to_laguerrel(norm2, n, alpha, beta);
    ft_triangular_bandedl * B = ft_create_B_laguerre_to_laguerrel(norm2, n, beta);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_laguerre_to_laguerre_diagonal_connection_coefficientl(norm1, norm2, n, alpha, beta, Vl, n+1);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++)
        V[i] = Vl[i];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    return V;
}

double * plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, const double alpha, const double beta, const double lambda) {
    double * V = plan_jacobi_to_jacobi(normjac, normultra, n, alpha, beta, lambda-0.5, lambda-0.5);
    if (normultra == 0) {
        double * sclrow = malloc(n*sizeof(double));
        if (n > 0)
            sclrow[0] = 1;
        for (int i = 1; i < n; i++)
            sclrow[i] = (lambda+i-0.5)/(2*lambda+i-1)*sclrow[i-1];
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= sclrow[i];
        free(sclrow);
    }
    return V;
}

double * plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, const double lambda, const double alpha, const double beta) {
    double * V = plan_jacobi_to_jacobi(normultra, normjac, n, lambda-0.5, lambda-0.5, alpha, beta);
    if (normultra == 0) {
        double * sclcol = malloc(n*sizeof(double));
        if (n > 0)
            sclcol[0] = 1;
        for (int i = 1; i < n; i++)
            sclcol[i] = (2*lambda+i-1)/(lambda+i-0.5)*sclcol[i-1];
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= sclcol[j];
        free(sclcol);
    }
    return V;
}

double * plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, const double alpha, const double beta) {
    double * V = plan_jacobi_to_jacobi(normjac, 1, n, alpha, beta, -0.5, -0.5);
    if (normcheb == 0) {
        double * sclrow = malloc(n*sizeof(double));
        for (int i = 0; i < n; i++)
            sclrow[i] = i ? sqrtl(M_2_PIl) : sqrtl(M_1_PIl);
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= sclrow[i];
        free(sclrow);
    }
    return V;
}

double * plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, const double alpha, const double beta) {
    double * V = plan_jacobi_to_jacobi(1, normjac, n, -0.5, -0.5, alpha, beta);
    if (normcheb == 0) {
        double * sclcol = malloc(n*sizeof(double));
        for (int i = 0; i < n; i++)
            sclcol[i] = i ? sqrtl(M_PI_2l) : sqrtl(M_PIl);
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= sclcol[j];
        free(sclcol);
    }
    return V;
}

double * plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, const double lambda) {
    double * V = plan_ultraspherical_to_jacobi(normultra, 1, n, lambda, -0.5, -0.5);
    if (normcheb == 0) {
        double * sclrow = malloc(n*sizeof(double));
        for (int i = 0; i < n; i++)
            sclrow[i] = i ? sqrtl(M_2_PIl) : sqrtl(M_1_PIl);
        for (int j = 0; j < n; j++)
            for (int i = j; i >= 0; i -= 2)
                V[i+j*n] *= sclrow[i];
        free(sclrow);
    }
    return V;
}

double * plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, const double lambda) {
    double * V = plan_jacobi_to_ultraspherical(1, normultra, n, -0.5, -0.5, lambda);
    if (normcheb == 0) {
        double * sclcol = malloc(n*sizeof(double));
        for (int i = 0; i < n; i++)
            sclcol[i] = i ? sqrtl(M_PI_2l) : sqrtl(M_PIl);
        for (int j = 0; j < n; j++)
            for (int i = j; i >= 0; i -= 2)
                V[i+j*n] *= sclcol[j];
        free(sclcol);
    }
    return V;
}

double * plan_associated_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const int c, const double alpha, const double beta, const double gamma, const double delta) {
    ft_triangular_bandedl * A = ft_create_A_associated_jacobi_to_jacobil(norm2, n, c, alpha, beta, gamma, delta);
    ft_triangular_bandedl * B = ft_create_B_associated_jacobi_to_jacobil(norm2, n, gamma, delta);
    ft_triangular_bandedl * C = ft_create_C_associated_jacobi_to_jacobil(norm2, n, gamma, delta);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_associated_jacobi_to_jacobi_diagonal_connection_coefficientl(norm1, norm2, n, c, alpha, beta, gamma, delta, Vl, n+1);
    ft_triangular_banded_quadratic_eigenvectorsl(A, B, C, Vl);
    double * V = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            V[i+j*n] = Vl[i+j*n];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    ft_destroy_triangular_bandedl(C);
    free(Vl);
    return V;
}

double * plan_associated_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const int c, const double alpha, const double beta) {
    ft_triangular_bandedl * A = ft_create_A_associated_laguerre_to_laguerrel(norm2, n, c, alpha, beta);
    ft_triangular_bandedl * B = ft_create_B_associated_laguerre_to_laguerrel(norm2, n, beta);
    ft_triangular_bandedl * C = ft_create_C_associated_laguerre_to_laguerrel(norm2, n, beta);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_associated_laguerre_to_laguerre_diagonal_connection_coefficientl(norm1, norm2, n, c, alpha, beta, Vl, n+1);
    ft_triangular_banded_quadratic_eigenvectorsl(A, B, C, Vl);
    double * V = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            V[i+j*n] = Vl[i+j*n];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    ft_destroy_triangular_bandedl(C);
    free(Vl);
    return V;
}

double * plan_associated_hermite_to_hermite(const int norm1, const int norm2, const int n, const int c) {
    ft_triangular_bandedl * A = ft_create_A_associated_hermite_to_hermitel(norm2, n, c);
    ft_triangular_bandedl * B = ft_create_B_associated_hermite_to_hermitel(norm2, n);
    ft_triangular_bandedl * C = ft_create_C_associated_hermite_to_hermitel(n);
    long double * Vl = calloc(n*n, sizeof(long double));
    ft_create_associated_hermite_to_hermite_diagonal_connection_coefficientl(norm1, norm2, n, c, Vl, n+1);
    ft_triangular_banded_quadratic_eigenvectorsl(A, B, C, Vl);
    double * V = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2)
            V[i+j*n] = Vl[i+j*n];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    ft_destroy_triangular_bandedl(C);
    free(Vl);
    return V;
}

#include "transforms_mpfr.c"
