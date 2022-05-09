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

void ft_destroy_modified_plan(ft_modified_plan * P) {
    if (P->nv < 1) {
        ft_destroy_triangular_banded(P->R);
    }
    else {
        ft_destroy_triangular_banded(P->K);
        ft_destroy_triangular_banded(P->R);
    }
    free(P);
}

void ft_mpmv(char TRANS, ft_modified_plan * P, double * x) {
    if (P->nv < 1) {
        ft_tbsv(TRANS, P->R, x);
    }
    else {
        if (TRANS == 'N') {
            ft_tbsv('N', P->K, x);
            ft_tbmv('N', P->R, x);
        }
        else if (TRANS == 'T') {
            ft_tbmv('T', P->R, x);
            ft_tbsv('T', P->K, x);
        }
    }
}

void ft_mpsv(char TRANS, ft_modified_plan * P, double * x) {
    if (P->nv < 1) {
        ft_tbmv(TRANS, P->R, x);
    }
    else {
        if (TRANS == 'N') {
            ft_tbsv('N', P->R, x);
            ft_tbmv('N', P->K, x);
        }
        else if (TRANS == 'T') {
            ft_tbmv('T', P->K, x);
            ft_tbsv('T', P->R, x);
        }
    }
}

void ft_mpmm(char TRANS, ft_modified_plan * P, double * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        ft_mpmv(TRANS, P, B+j*LDB);
}

void ft_mpsm(char TRANS, ft_modified_plan * P, double * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        ft_mpsv(TRANS, P, B+j*LDB);
}

ft_modified_plan * ft_plan_modified_jacobi_to_jacobi(const int n, const double alpha, const double beta, const int nu, const double * u, const int nv, const double * v, const int verbose) {
    if (nv < 1) {
        // polynomial case
        ft_banded * U = ft_operator_normalized_jacobi_clenshaw(n, nu, u, 1, alpha, beta);
        ft_banded_cholfact(U);
        ft_triangular_banded * R = ft_convert_banded_to_triangular_banded(U);
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->R = R;
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        return P;
    }
    else {
        // rational case
        ft_banded_ql * F;
        int N = 2*n;
        while (1) {
            if (N > FT_MODIFIED_NMAX) exit_failure("ft_plan_modified_jacobi_to_jacobi: dimension of QL factorization, N, exceeds maximum allowable.");
            ft_banded * V = ft_operator_normalized_jacobi_clenshaw(N+nu+nv, nv, v, 1, alpha, beta);

            double nrm_Vb = 0;
            double * Vb = calloc(N*(nv-1), sizeof(double));
            for (int j = 0; j < nv-1; j++)
                for (int i = N-nv+1+j; i < N; i++) {
                    Vb[i+j*N] = ft_get_banded_index(V, i, j+N);
                    nrm_Vb += Vb[i+j*N]*Vb[i+j*N];
                }
            nrm_Vb = sqrt(nrm_Vb);

            // truncate it for QL
            V->m = V->n = N;
            F = ft_banded_qlfact(V);

            for (int j = 0; j < nv-1; j++)
                ft_bqmv('T', F, Vb+j*N);

            double nrm_Vn = 0;
            for (int j = 0; j < nv-1; j++)
                for (int i = 0; i < n; i++)
                    nrm_Vn += Vb[i+j*N]*Vb[i+j*N];
            nrm_Vn = sqrt(nrm_Vn);

            free(Vb);
            ft_destroy_banded(V);
            //if (nv*nrm_Vn <= eps()*nrm_Vb) break;
            if (nv*nrm_Vn <= eps()*nrm_Vb) {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %17.16e ≤ %17.16e\n", N, nv*nrm_Vn, eps()*nrm_Vb);
                break;
            }
            else {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %17.16e ≰ %17.16e\n", N, nv*nrm_Vn, eps()*nrm_Vb);
            }
            ft_destroy_banded_ql(F);
            N <<= 1;
        }
        F->factors->m = F->factors->n = n+nu+nv;
        ft_banded * U = ft_operator_normalized_jacobi_clenshaw(n+nu+nv, nu, u, 1, alpha, beta);

        ft_banded * Lt = ft_calloc_banded(n+nu+nv, n+nu+nv, 0, F->factors->l);
        for (int j = 0; j < n+nu+nv; j++)
            for (int i = j; i < MIN(n+nu+nv, j+F->factors->l+1); i++)
                ft_set_banded_index(Lt, ft_get_banded_index(F->factors, i, j), j, i);

        double * D = calloc(n+nu+nv, sizeof(double));
        for (int j = 0; j < n+nu+nv; j++) {
            D[j] = (signbit(ft_get_banded_index(Lt, j, j))) ? -1 : 1;
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                ft_set_banded_index(Lt, ft_get_banded_index(Lt, i, j)*D[j], i, j);
        }

        ft_banded * ULt = ft_calloc_banded(n+nu+nv, n+nu+nv, nu+nv-2, nu+2*nv-3);
        ft_gbmm(1, U, Lt, 0, ULt);
        // ULᵀ ← QᵀULᵀ
        ft_partial_bqmm(F, nu, nv, ULt);

        int b = nu+nv-2;
        ft_banded * QtULt = ft_calloc_banded(n, n, b, b);
        for (int i = 0; i < n; i++)
            for (int j = MAX(i-b, 0); j < MIN(i+b+1, n); j++)
                ft_set_banded_index(QtULt, D[i]*ft_get_banded_index(ULt, i, j), i, j);
        ft_banded_cholfact(QtULt);
        ft_triangular_banded * K = ft_convert_banded_to_triangular_banded(QtULt);

        ft_triangular_banded * R = ft_calloc_triangular_banded(n, Lt->u);
        for (int j = 0; j < n; j++)
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                ft_set_triangular_banded_index(R, ft_get_banded_index(Lt, i, j), i, j);

        free(D);
        ft_destroy_banded(U);
        ft_destroy_banded(Lt);
        ft_destroy_banded(ULt);
        ft_destroy_banded_ql(F);
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        P->K = K;
        P->R = R;
        return P;
    }
}

ft_modified_plan * ft_plan_modified_laguerre_to_laguerre(const int n, const double alpha, const int nu, const double * u, const int nv, const double * v, const int verbose) {
    if (nv < 1) {
        // polynomial case
        ft_banded * U = ft_operator_normalized_laguerre_clenshaw(n, nu, u, 1, alpha);
        ft_banded_cholfact(U);
        ft_triangular_banded * R = ft_convert_banded_to_triangular_banded(U);
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->R = R;
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        return P;
    }
    else {
        // rational case
        ft_banded_ql * F;
        int N = 2*n;
        while (1) {
            if (N > FT_MODIFIED_NMAX) exit_failure("ft_plan_modified_laguerre_to_laguerre: dimension of QL factorization, N, exceeds maximum allowable.");
            ft_banded * V = ft_operator_normalized_laguerre_clenshaw(N+nu+nv, nv, v, 1, alpha);

            double nrm_Vb = 0;
            double * Vb = calloc(N*(nv-1), sizeof(double));
            for (int j = 0; j < nv-1; j++)
                for (int i = N-nv+1+j; i < N; i++) {
                    Vb[i+j*N] = ft_get_banded_index(V, i, j+N);
                    nrm_Vb += Vb[i+j*N]*Vb[i+j*N];
                }
            nrm_Vb = sqrt(nrm_Vb);

            // truncate it for QL
            V->m = V->n = N;
            F = ft_banded_qlfact(V);

            for (int j = 0; j < nv-1; j++)
                ft_bqmv('T', F, Vb+j*N);

            double nrm_Vn = 0;
            for (int j = 0; j < nv-1; j++)
                for (int i = 0; i < n; i++)
                    nrm_Vn += Vb[i+j*N]*Vb[i+j*N];
            nrm_Vn = sqrt(nrm_Vn);

            free(Vb);
            ft_destroy_banded(V);
            //if (nv*nrm_Vn <= eps()*nrm_Vb) break;
            if (nv*nrm_Vn <= eps()*nrm_Vb) {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %17.16e ≤ %17.16e\n", N, nv*nrm_Vn, eps()*nrm_Vb);
                break;
            }
            else {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %17.16e ≰ %17.16e\n", N, nv*nrm_Vn, eps()*nrm_Vb);
            }
            ft_destroy_banded_ql(F);
            N <<= 1;
        }
        F->factors->m = F->factors->n = n+nu+nv;
        ft_banded * U = ft_operator_normalized_laguerre_clenshaw(n+nu+nv, nu, u, 1, alpha);

        ft_banded * Lt = ft_calloc_banded(n+nu+nv, n+nu+nv, 0, F->factors->l);
        for (int j = 0; j < n+nu+nv; j++)
            for (int i = j; i < MIN(n+nu+nv, j+F->factors->l+1); i++)
                ft_set_banded_index(Lt, ft_get_banded_index(F->factors, i, j), j, i);

        double * D = calloc(n+nu+nv, sizeof(double));
        for (int j = 0; j < n+nu+nv; j++) {
            D[j] = (signbit(ft_get_banded_index(Lt, j, j))) ? -1 : 1;
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                ft_set_banded_index(Lt, ft_get_banded_index(Lt, i, j)*D[j], i, j);
        }

        ft_banded * ULt = ft_calloc_banded(n+nu+nv, n+nu+nv, nu+nv-2, nu+2*nv-3);
        ft_gbmm(1, U, Lt, 0, ULt);
        // ULᵀ ← QᵀULᵀ
        ft_partial_bqmm(F, nu, nv, ULt);

        int b = nu+nv-2;
        ft_banded * QtULt = ft_calloc_banded(n, n, b, b);
        for (int i = 0; i < n; i++)
            for (int j = MAX(i-b, 0); j < MIN(i+b+1, n); j++)
                ft_set_banded_index(QtULt, D[i]*ft_get_banded_index(ULt, i, j), i, j);
        ft_banded_cholfact(QtULt);
        ft_triangular_banded * K = ft_convert_banded_to_triangular_banded(QtULt);

        ft_triangular_banded * R = ft_calloc_triangular_banded(n, Lt->u);
        for (int j = 0; j < n; j++)
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                ft_set_triangular_banded_index(R, ft_get_banded_index(Lt, i, j), i, j);

        free(D);
        ft_destroy_banded(U);
        ft_destroy_banded(Lt);
        ft_destroy_banded(ULt);
        ft_destroy_banded_ql(F);
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        P->K = K;
        P->R = R;
        return P;
    }
}

ft_modified_plan * ft_plan_modified_hermite_to_hermite(const int n, const int nu, const double * u, const int nv, const double * v, const int verbose) {
    if (nv < 1) {
        // polynomial case
        ft_banded * U = ft_operator_normalized_hermite_clenshaw(n, nu, u, 1);
        ft_banded_cholfact(U);
        ft_triangular_banded * R = ft_convert_banded_to_triangular_banded(U);
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->R = R;
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        return P;
    }
    else {
        // rational case
        ft_banded_ql * F;
        int N = 2*n;
        while (1) {
            if (N > FT_MODIFIED_NMAX) exit_failure("ft_plan_modified_hermite_to_hermite: dimension of QL factorization, N, exceeds maximum allowable.");
            ft_banded * V = ft_operator_normalized_hermite_clenshaw(N+nu+nv, nv, v, 1);

            double nrm_Vb = 0;
            double * Vb = calloc(N*(nv-1), sizeof(double));
            for (int j = 0; j < nv-1; j++)
                for (int i = N-nv+1+j; i < N; i++) {
                    Vb[i+j*N] = ft_get_banded_index(V, i, j+N);
                    nrm_Vb += Vb[i+j*N]*Vb[i+j*N];
                }
            nrm_Vb = sqrt(nrm_Vb);

            // truncate it for QL
            V->m = V->n = N;
            F = ft_banded_qlfact(V);

            for (int j = 0; j < nv-1; j++)
                ft_bqmv('T', F, Vb+j*N);

            double nrm_Vn = 0;
            for (int j = 0; j < nv-1; j++)
                for (int i = 0; i < n; i++)
                    nrm_Vn += Vb[i+j*N]*Vb[i+j*N];
            nrm_Vn = sqrt(nrm_Vn);

            free(Vb);
            ft_destroy_banded(V);
            //if (nv*nrm_Vn <= eps()*nrm_Vb) break;
            if (nv*nrm_Vn <= eps()*nrm_Vb) {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %17.16e ≤ %17.16e\n", N, nv*nrm_Vn, eps()*nrm_Vb);
                break;
            }
            else {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %17.16e ≰ %17.16e\n", N, nv*nrm_Vn, eps()*nrm_Vb);
            }
            ft_destroy_banded_ql(F);
            N <<= 1;
        }
        F->factors->m = F->factors->n = n+nu+nv;
        ft_banded * U = ft_operator_normalized_hermite_clenshaw(n+nu+nv, nu, u, 1);

        ft_banded * Lt = ft_calloc_banded(n+nu+nv, n+nu+nv, 0, F->factors->l);
        for (int j = 0; j < n+nu+nv; j++)
            for (int i = j; i < MIN(n+nu+nv, j+F->factors->l+1); i++)
                ft_set_banded_index(Lt, ft_get_banded_index(F->factors, i, j), j, i);

        double * D = calloc(n+nu+nv, sizeof(double));
        for (int j = 0; j < n+nu+nv; j++) {
            D[j] = (signbit(ft_get_banded_index(Lt, j, j))) ? -1 : 1;
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                ft_set_banded_index(Lt, ft_get_banded_index(Lt, i, j)*D[j], i, j);
        }

        ft_banded * ULt = ft_calloc_banded(n+nu+nv, n+nu+nv, nu+nv-2, nu+2*nv-3);
        ft_gbmm(1, U, Lt, 0, ULt);
        // ULᵀ ← QᵀULᵀ
        ft_partial_bqmm(F, nu, nv, ULt);

        int b = nu+nv-2;
        ft_banded * QtULt = ft_calloc_banded(n, n, b, b);
        for (int i = 0; i < n; i++)
            for (int j = MAX(i-b, 0); j < MIN(i+b+1, n); j++)
                ft_set_banded_index(QtULt, D[i]*ft_get_banded_index(ULt, i, j), i, j);
        ft_banded_cholfact(QtULt);
        ft_triangular_banded * K = ft_convert_banded_to_triangular_banded(QtULt);

        ft_triangular_banded * R = ft_calloc_triangular_banded(n, Lt->u);
        for (int j = 0; j < n; j++)
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                ft_set_triangular_banded_index(R, ft_get_banded_index(Lt, i, j), i, j);

        free(D);
        ft_destroy_banded(U);
        ft_destroy_banded(Lt);
        ft_destroy_banded(ULt);
        ft_destroy_banded_ql(F);
        ft_modified_plan * P = malloc(sizeof(ft_modified_plan));
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        P->K = K;
        P->R = R;
        return P;
    }
}

#include "transforms_mpfr.c"
