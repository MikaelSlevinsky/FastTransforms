// Computational routines for one-dimensional orthogonal polynomial transforms.

#include "fasttransforms.h"
#include "ftinternal.h"

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
    ft_triangular_bandedl * A = ft_create_A_legendre_to_chebyshevl(n);
    ft_triangular_bandedl * B = ft_create_B_legendre_to_chebyshevl(n);
    long double * Vl = calloc(n*n, sizeof(long double));
    if (n > 0)
        Vl[0] = 1;
    if (n > 1)
        Vl[1+n] = 1;
    for (int i = 2; i < n; i++)
        Vl[i+i*n] = (2*i-1)*Vl[i-1+(i-1)*n]/(2*i);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = calloc(n*n, sizeof(double));
    long double * sclrow = malloc(n*sizeof(long double));
    long double * sclcol = malloc(n*sizeof(long double));
    for (int i = 0; i < n; i++) {
        sclrow[i] = normcheb ? i ? sqrtl(M_PI_2l) : sqrtl(M_PIl) : 1.0L;
        sclcol[i] = normleg ? sqrtl(i+0.5L) : 1.0L;
    }
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2)
            V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    free(sclrow);
    free(sclcol);
    return V;
}

double * plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n) {
    ft_triangular_bandedl * A = ft_create_A_chebyshev_to_legendrel(n);
    ft_triangular_bandedl * B = ft_create_B_chebyshev_to_legendrel(n);
    long double * Vl = calloc(n*n, sizeof(long double));
    if (n > 0)
        Vl[0] = 1;
    if (n > 1)
        Vl[1+n] = 1;
    for (int i = 2; i < n; i++)
        Vl[i+i*n] = (2*i)*Vl[i-1+(i-1)*n]/(2*i-1);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = calloc(n*n, sizeof(double));
    long double * sclrow = malloc(n*sizeof(long double));
    long double * sclcol = malloc(n*sizeof(long double));
    for (int i = 0; i < n; i++) {
        sclrow[i] = normleg ? 1.0L/sqrtl(i+0.5L) : 1.0L;
        sclcol[i] = normcheb ? i ? sqrtl(M_2_PIl) : sqrtl(M_1_PIl) : 1.0L;
    }
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2)
            V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    free(sclrow);
    free(sclcol);
    return V;
}

double * plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, const double lambda, const double mu) {
    ft_triangular_bandedl * A = ft_create_A_ultraspherical_to_ultrasphericall(n, lambda, mu);
    ft_triangular_bandedl * B = ft_create_B_ultraspherical_to_ultrasphericall(n, mu);
    long double * Vl = calloc(n*n, sizeof(long double));
    long double lambdal = lambda, mul = mu;
    if (n > 0)
        Vl[0] = 1;
    for (int i = 1; i < n; i++)
        Vl[i+i*n] = (i-1+lambdal)*Vl[i-1+(i-1)*n]/(i-1+mul);
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = calloc(n*n, sizeof(double));
    long double * sclrow = calloc(n, sizeof(long double));
    long double * sclcol = calloc(n, sizeof(long double));
    if (n > 0) {
        sclrow[0] = norm2 ? sqrtl(sqrtl(M_PIl)*tgammal(mul+0.5L)/tgammal(mul+1)) : 1.0L;
        sclcol[0] = norm1 ? sqrtl(tgammal(lambdal+1)/(sqrtl(M_PIl)*tgammal(lambdal+0.5L))) : 1.0L;
    }
    for (int i = 1; i < n; i++) {
        sclrow[i] = norm2 ? sqrtl((i-1+mul)/i*(i-1+2*mul)/(i+mul))*sclrow[i-1] : 1.0L;
        sclcol[i] = norm1 ? sqrtl(i/(i-1+lambdal)*(i+lambdal)/(i-1+2*lambdal))*sclcol[i-1] : 1.0L;
    }
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2)
            V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    free(sclrow);
    free(sclcol);
    return V;
}

double * plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const double alpha, const double beta, const double gamma, const double delta) {
    ft_triangular_bandedl * A = ft_create_A_jacobi_to_jacobil(n, alpha, beta, gamma, delta);
    ft_triangular_bandedl * B = ft_create_B_jacobi_to_jacobil(n, gamma, delta);
    long double alphal = alpha, betal = beta, gammal = gamma, deltal = delta;
    long double * Vl = calloc(n*n, sizeof(long double));
    if (n > 0)
        Vl[0] = 1;
    if (n > 1)
        Vl[1+n] = (alphal+betal+2)/(gammal+deltal+2);
    for (int i = 2; i < n; i++)
        Vl[i+i*n] = (2*i+alphal+betal-1)/(i+alphal+betal)*(2*i+alphal+betal)/(2*i+gammal+deltal-1)*(i+gammal+deltal)/(2*i+gammal+deltal)*Vl[i-1+(i-1)*n];
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = calloc(n*n, sizeof(double));
    long double * sclrow = calloc(n, sizeof(long double));
    long double * sclcol = calloc(n, sizeof(long double));
    if (n > 0) {
        sclrow[0] = norm2 ? sqrtl(powl(2.0L, gammal+deltal+1)*tgammal(gammal+1)*tgammal(deltal+1)/tgammal(gammal+deltal+2)) : 1.0L;
        sclcol[0] = norm1 ? sqrtl(tgammal(alphal+betal+2)/(powl(2.0L, alphal+betal+1)*tgammal(alphal+1)*tgammal(betal+1))) : 1.0L;
    }
    if (n > 1) {
        sclrow[1] = norm2 ? sqrtl((gammal+1)*(deltal+1)/(gammal+deltal+3))*sclrow[0] : 1.0L;
        sclcol[1] = norm1 ? sqrtl((alphal+betal+3)/(alphal+1)/(betal+1))*sclcol[0] : 1.0L;
    }
    for (int i = 2; i < n; i++) {
        sclrow[i] = norm2 ? sqrtl((i+gammal)/i*(i+deltal)/(i+gammal+deltal)*(2*i+gammal+deltal-1)/(2*i+gammal+deltal+1))*sclrow[i-1] : 1.0L;
        sclcol[i] = norm1 ? sqrtl(i/(i+alphal)*(i+alphal+betal)/(i+betal)*(2*i+alphal+betal+1)/(2*i+alphal+betal-1))*sclcol[i-1] : 1.0L;
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    free(sclrow);
    free(sclcol);
    return V;
}

double * plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const double alpha, const double beta) {
    ft_triangular_bandedl * A = ft_create_A_laguerre_to_laguerrel(n, alpha, beta);
    ft_triangular_bandedl * B = ft_create_B_laguerre_to_laguerrel(n);
    long double * Vl = calloc(n*n, sizeof(long double));
    long double alphal = alpha, betal = beta;
    for (int i = 0; i < n; i++)
        Vl[i+i*n] = 1;
    ft_triangular_banded_eigenvectorsl(A, B, Vl);
    double * V = calloc(n*n, sizeof(double));
    long double * sclrow = calloc(n, sizeof(long double));
    long double * sclcol = calloc(n, sizeof(long double));
    if (n > 0) {
        sclrow[0] = norm2 ? sqrtl(tgammal(betal+1)) : 1.0L;
        sclcol[0] = norm1 ? 1.0L/sqrtl(tgammal(alphal+1)) : 1.0L;
    }
    for (int i = 1; i < n; i++) {
        sclrow[i] = norm2 ? sqrtl((i+betal)/i)*sclrow[i-1] : 1.0L;
        sclcol[i] = norm1 ? sqrtl(i/(i+alphal))*sclcol[i-1] : 1.0L;
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    free(Vl);
    free(sclrow);
    free(sclcol);
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

double * plan_associated_jacobi_to_jacobi(const int norm2, const int n, const int c, const double alpha, const double beta, const double gamma, const double delta) {
    ft_triangular_bandedl * A = ft_create_A_associated_jacobi_to_jacobil(n, alpha, beta, gamma, delta);
    ft_triangular_bandedl * B = ft_create_B_associated_jacobi_to_jacobil(n, gamma, delta);
    ft_triangular_bandedl * C = ft_create_C_associated_jacobi_to_jacobil(n, gamma, delta);
    long double alphal = alpha, betal = beta, gammal = gamma, deltal = delta;
    long double * lambdal = malloc(n*sizeof(long double));
    for (int j = 0; j < n; j++)
        lambdal[j] = (j+alphal+betal+2*c-1)*(j+alphal+betal+2*c+1) + (j+3)*(j-1.0L);
    long double * Vl = calloc(n*n, sizeof(long double));
    if (n > 0)
        Vl[0] = 1;
    if (n > 1)
        Vl[1+n] = (2*c+alphal+betal+1)/(c+alphal+betal+1)*(2*c+alphal+betal+2)/(1+c)/(gammal+deltal+2);
    for (int i = 2; i < n; i++)
        Vl[i+i*n] = (2*(i+c)+alphal+betal-1)/(i+c+alphal+betal)*(2*(i+c)+alphal+betal)/(2*i+gammal+deltal-1)*(i+gammal+deltal)/(2*i+gammal+deltal)*i/(i+c)*Vl[i-1+(i-1)*n];
    ft_triangular_banded_eigenvectors_3argl(A, B, lambdal, C, Vl);
    double * V = calloc(n*n, sizeof(double));
    long double * sclrow = calloc(n, sizeof(long double));
    if (n > 0)
        sclrow[0] = norm2 ? sqrtl(powl(2.0L, gammal+deltal+1)*tgammal(gammal+1)*tgammal(deltal+1)/tgammal(gammal+deltal+2)) : 1.0L;
    if (n > 1)
        sclrow[1] = norm2 ? sqrtl((gammal+1)*(deltal+1)/(gammal+deltal+3))*sclrow[0] : 1.0L;
    for (int i = 2; i < n; i++)
        sclrow[i] = norm2 ? sqrtl((i+gammal)/i*(i+deltal)/(i+gammal+deltal)*(2*i+gammal+deltal-1)/(2*i+gammal+deltal+1))*sclrow[i-1] : 1.0L;
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            V[i+j*n] = sclrow[i]*Vl[i+j*n];
    ft_destroy_triangular_bandedl(A);
    ft_destroy_triangular_bandedl(B);
    ft_destroy_triangular_bandedl(C);
    free(Vl);
    free(sclrow);
    return V;
}

#include "transforms_mpfr.c"
