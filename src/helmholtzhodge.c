#include "fasttransforms.h"
#include "ftinternal.h"

static inline ft_triangular_banded * sph_sine_multiplication(const int n, const int m) {
    ft_triangular_banded * T = ft_calloc_triangular_banded(n+1-m, 2);
    double num, den, cst;
    for (int l = 0; l < n+1-m; l++) {
        num = (l+2*m-1)*(l+2*m);
        den = (2*l+2*m-1)*(2*l+2*m+1);
        cst = sqrt(num/den);
        ft_set_triangular_banded_index(T, cst, l, l);
    }
    for (int l = 0; l < n-1-m; l++) {
        num = (l+1)*(l+2);
        den = (2*l+2*m+1)*(2*l+2*m+3);
        cst = -sqrt(num/den);
        ft_set_triangular_banded_index(T, cst, l, l+2);
    }
    return T;
}

static inline ft_banded * sph_theta_derivative(const int n, const int m) {
    ft_banded * B = ft_calloc_banded(n+1-m, n+1-m, 1, 1);
    double num, den, cst;
    for (int l = 1; l < n+1-m; l++) {
        num = l*(l+2*m);
        den = (2*l+2*m-1)*(2*l+2*m+1);
        cst = (l+m-1)*sqrt(num/den);
        ft_set_banded_index(B, cst, l, l-1);
    }
    for (int l = 0; l < n-m; l++) {
        num = (l+1)*(l+2*m+1);
        den = (2*l+2*m+1)*(2*l+2*m+3);
        cst = -(l+m+2)*sqrt(num/den);
        ft_set_banded_index(B, cst, l, l+1);
    }
    return B;
}

void ft_destroy_gradient_plan(ft_gradient_plan * P) {
    for (int m = 0; m < P->n; m++) {
        ft_destroy_banded(P->B[m]);
        ft_destroy_triangular_banded(P->T[m]);
    }
	free(P->B);
    free(P->T);
    free(P);
}

ft_gradient_plan * ft_plan_sph_gradient(const int n) {
	ft_banded ** B = malloc(n*sizeof(ft_banded *));
	ft_triangular_banded ** T = malloc(n*sizeof(ft_triangular_banded *));
	for (int m = 1; m <= n; m++) {
		B[m-1] = sph_theta_derivative(n, m);
		T[m-1] = sph_sine_multiplication(n, m);
	}
	ft_gradient_plan * P = malloc(sizeof(ft_gradient_plan));
	P->B = B;
	P->T = T;
	P->n = n;
	return P;
}

void ft_execute_sph_gradient(ft_gradient_plan * P, double * U, double * Ut, double * Up, const int N, const int M) {
	ft_banded ** B = P->B;
	ft_triangular_banded ** T = P->T;
	for (int l = 0; l < N-1; l++) {
		Ut[l] = -sqrt((l+1.0)*(l+2.0))*U[l+1];
		Up[l] = 0.0;
	}
	Ut[N-1] = Up[N-1] = 0.0;
	for (int m = 1; m <= M/2; m++) {
		ft_gbmv(1, B[m-1], U+N*(2*m-1), 0, Ut+N*(2*m-1));
		ft_gbmv(1, B[m-1], U+N*(2*m), 0, Ut+N*(2*m));
		ft_tbsv('N', T[m-1], Ut+N*(2*m-1));
		ft_tbsv('N', T[m-1], Ut+N*(2*m));
	}
	for (int m = 1; m <= M/2; m++) {
		for (int l = 0; l < N+1-m; l++) {
			Up[l+N*(2*m-1)] = -m*U[l+N*(2*m)];
			Up[l+N*(2*m)] = m*U[l+N*(2*m-1)];
		}
		ft_tbsv('N', T[m-1], Up+N*(2*m-1));
		ft_tbsv('N', T[m-1], Up+N*(2*m));
	}
}

void ft_execute_sph_curl(ft_gradient_plan * P, double * U, double * Ut, double * Up, const int N, const int M) {
	ft_execute_sph_gradient(P, U, Up, Ut, N, M);
	for (int i = 0; i < N*M; i++)
		Ut[i] = -Ut[i];
}

void ft_destroy_helmholtzhodge_plan(ft_helmholtzhodge_plan * P) {
    for (int m = 0; m < P->n; m++) {
        ft_destroy_triangular_banded(P->T[m]);
        ft_destroy_banded_qr(P->F[m]);
    }
    free(P->T);
    free(P->F);
    free(P->X);
    free(P);
}

static inline ft_banded * sph_helmholtz_hodge_conversion(const int n, const int m) {
    ft_banded * A = ft_calloc_banded(2*n+4-2*m, 2*n+2-2*m, 2, 2);
    double num, den, cst;
    for (int l = 0; l < n+1-m; l++) {
        ft_set_banded_index(A, m, 2*l+1, 2*l);
        ft_set_banded_index(A, m, 2*l, 2*l+1);
        num = (l+1)*(l+2*m+1);
        den = (2*l+2*m+1)*(2*l+2*m+3);
        cst = (m+l)*sqrt(num/den);
        ft_set_banded_index(A, cst, 2*l+3, 2*l+1);
        ft_set_banded_index(A, cst, 2*l+2, 2*l);
    }
    for (int l = 0; l < n-m; l++) {
        num = (l+1)*(l+2*m+1);
        den = (2*l+2*m+1)*(2*l+2*m+3);
        cst = -(m+l+2)*sqrt(num/den);
        ft_set_banded_index(A, cst, 2*l, 2*l+2);
        ft_set_banded_index(A, cst, 2*l+1, 2*l+3);
    }
    return A;
}

ft_helmholtzhodge_plan * ft_plan_sph_helmholtzhodge(const int n) {
    ft_triangular_banded ** T = malloc(n*sizeof(ft_triangular_banded *));
    ft_banded_qr ** F = malloc(n*sizeof(ft_banded_qr *));
    for (int m = 1; m <= n; m++) {
        T[m-1] = sph_sine_multiplication(n, m);
        ft_banded * A = sph_helmholtz_hodge_conversion(n, m);
        F[m-1] = ft_banded_qrfact(A);
        ft_destroy_banded(A);
    }
    ft_helmholtzhodge_plan * P = malloc(sizeof(ft_helmholtzhodge_plan));
    P->T = T;
    P->F = F;
    P->X = malloc((2*n+2)*sizeof(double));
    P->n = n;
    return P;
}

static inline void sph_hh_readin1(double * X, double * V1, double * V2, const int N, const int m) {
	X[0] = V1[N*(2*m-1)];
	X[2*N+1-2*m] = V2[N-m+N*(2*m)];
	for (int l = 0; l < N-m; l++) {
		X[2*l+1] = V2[l+N*(2*m)];
		X[2*l+2] = V1[l+1+N*(2*m-1)];
	}
	X[2*N+2-2*m] = X[2*N+3-2*m] = 0;
}

static inline void sph_hh_readin2(double * X, double * V1, double * V2, const int N, const int m) {
	X[0] = V1[N*(2*m)];
	X[2*N+1-2*m] = -V2[N-m+N*(2*m-1)];
	for (int l = 0; l < N-m; l++) {
		X[2*l+1] = -V2[l+N*(2*m-1)];
		X[2*l+2] = V1[l+1+N*(2*m)];
	}
	X[2*N+2-2*m] = X[2*N+3-2*m] = 0;
}

static inline void sph_hh_writeout1(double * X, double * U1, double * U2, const int N, const int m) {
	U1[N*(2*m-1)] = X[0];
	U2[N*(2*m)] = X[N-m];
	for (int l = 0; l < N-1-m; l++) {
		U1[l+1+N*(2*m-1)] = X[2*l+2];
		U2[l+N*(2*m)] = X[2*l+1];
	}
	U2[N-1-m+N*(2*m)] = X[2*N-2*m-1];
}

static inline void sph_hh_writeout2(double * X, double * U1, double * U2, const int N, const int m) {
	U1[N*(2*m)] = X[0];
	U2[N*(2*m-1)] = X[N-m];
	for (int l = 0; l < N-1-m; l++) {
		U1[l+1+N*(2*m)] = X[2*l+2];
		U2[l+N*(2*m-1)] = -X[2*l+1];
	}
	U2[N-1-m+N*(2*m-1)] = -X[2*N-2*m-1];
}

// Separates a vector field V1 e_θ + V2 e_φ into ∇U1 + e_r × ∇U2.
// V1 and V2 can be synthesized and analyzed via sphv2fourier, while
// U1 and U2 are expanded in spherical harmonics.
void ft_execute_sph_helmholtzhodge(ft_helmholtzhodge_plan * P, double * U1, double * U2, double * V1, double * V2, const int N, const int M) {
    ft_triangular_banded ** T = P->T;
    ft_banded_qr ** F = P->F;
    double * X = P->X;
    U1[0] = U2[0] = 0;
    for (int l = 0; l < N-1; l++) {
        double den = sqrt((l+1.0)*(l+2.0));
        U1[l+1] = -V1[l]/den;
        U2[l+1] = -V2[l]/den;
    }
    for (int m = 1; m <= M/2; m++) {
        ft_tbmv('N', T[m-1], V1+N*(2*m-1));
        ft_tbmv('N', T[m-1], V1+N*(2*m));
        ft_tbmv('N', T[m-1], V2+N*(2*m-1));
        ft_tbmv('N', T[m-1], V2+N*(2*m));
        sph_hh_readin1(X, V1, V2, N, m);
		ft_bqmv('T', F[m-1], X);
		ft_brsv('N', F[m-1], X);
        sph_hh_writeout1(X, U1, U2, N, m);
        sph_hh_readin2(X, V1, V2, N, m);
		ft_bqmv('T', F[m-1], X);
		ft_brsv('N', F[m-1], X);
        sph_hh_writeout2(X, U1, U2, N, m);
    }
}
