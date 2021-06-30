#include "fasttransforms.h"
#include "ftinternal.h"

static inline void compute_givens(double x, double y, double * c, double * s, double * r) {
    * r = hypot(x, y);
    if (* r <= M_FLT_MIN) {
        * c = 1.0;
        * s = 0.0;
    }
    else {
        * c = x / * r;
        * s = y / * r;
    }
}

ft_ZYZR ft_create_ZYZR(ft_orthogonal_transformation Q) {
    double s[3], c[3], r, t1, t2;
    compute_givens(Q.Q[6], Q.Q[7], &c[0], &s[0], &r);
    t1 = c[0]*Q.Q[4] - s[0]*Q.Q[3];
    if (t1 < 0.0) {
        c[0] = -c[0];
        s[0] = -s[0];
        r = -r;
    }
    t1 = c[0]*Q.Q[0] + s[0]*Q.Q[1];
    t2 = c[0]*Q.Q[1] - s[0]*Q.Q[0];
    Q.Q[0] = t1;
    Q.Q[1] = t2;
    t1 = c[0]*Q.Q[3] + s[0]*Q.Q[4];
    t2 = c[0]*Q.Q[4] - s[0]*Q.Q[3];
    Q.Q[3] = t1;
    Q.Q[4] = t2;
    Q.Q[6] = r;
    Q.Q[7] = 0.0;

    compute_givens(Q.Q[8], Q.Q[6], &c[1], &s[1], &r);
    t1 = c[1]*Q.Q[0] - s[1]*Q.Q[2];
    if (t1 < 0.0) {
        c[1] = -c[1];
        s[1] = -s[1];
        r = -r;
    }
    t1 = c[1]*Q.Q[2] + s[1]*Q.Q[0];
    t2 = c[1]*Q.Q[0] - s[1]*Q.Q[2];
    Q.Q[2] = t1;
    Q.Q[0] = t2;
    t1 = c[1]*Q.Q[5] + s[1]*Q.Q[3];
    t2 = c[1]*Q.Q[3] - s[1]*Q.Q[5];
    Q.Q[5] = t1;
    Q.Q[3] = t2;
    Q.Q[8] = r;
    Q.Q[6] = 0.0;
    compute_givens(Q.Q[4], Q.Q[3], &c[2], &s[2], &r);
    int sign = Q.Q[8] > 0 ? 1 : -1;
    return (ft_ZYZR) {{s[0], -s[1], -s[2]}, {c[0], c[1], c[2]}, sign};
}

void ft_apply_ZYZR(ft_ZYZR Q, ft_orthogonal_transformation * U) {
    double t1, t2;

    if (Q.sign < 0) {
        U->Q[2] = -U->Q[2];
        U->Q[5] = -U->Q[5];
        U->Q[8] = -U->Q[8];
    }

    t1 = Q.c[2]*U->Q[0] - Q.s[2]*U->Q[1];
    t2 = Q.c[2]*U->Q[1] + Q.s[2]*U->Q[0];
    U->Q[0] = t1;
    U->Q[1] = t2;
    t1 = Q.c[2]*U->Q[3] - Q.s[2]*U->Q[4];
    t2 = Q.c[2]*U->Q[4] + Q.s[2]*U->Q[3];
    U->Q[3] = t1;
    U->Q[4] = t2;
    t1 = Q.c[2]*U->Q[6] - Q.s[2]*U->Q[7];
    t2 = Q.c[2]*U->Q[7] + Q.s[2]*U->Q[6];
    U->Q[6] = t1;
    U->Q[7] = t2;

    t1 = Q.c[1]*U->Q[0] - Q.s[1]*U->Q[2];
    t2 = Q.c[1]*U->Q[2] + Q.s[1]*U->Q[0];
    U->Q[0] = t1;
    U->Q[2] = t2;
    t1 = Q.c[1]*U->Q[3] - Q.s[1]*U->Q[5];
    t2 = Q.c[1]*U->Q[5] + Q.s[1]*U->Q[3];
    U->Q[3] = t1;
    U->Q[5] = t2;
    t1 = Q.c[1]*U->Q[6] - Q.s[1]*U->Q[8];
    t2 = Q.c[1]*U->Q[8] + Q.s[1]*U->Q[6];
    U->Q[6] = t1;
    U->Q[8] = t2;

    t1 = Q.c[0]*U->Q[0] - Q.s[0]*U->Q[1];
    t2 = Q.c[0]*U->Q[1] + Q.s[0]*U->Q[0];
    U->Q[0] = t1;
    U->Q[1] = t2;
    t1 = Q.c[0]*U->Q[3] - Q.s[0]*U->Q[4];
    t2 = Q.c[0]*U->Q[4] + Q.s[0]*U->Q[3];
    U->Q[3] = t1;
    U->Q[4] = t2;
    t1 = Q.c[0]*U->Q[6] - Q.s[0]*U->Q[7];
    t2 = Q.c[0]*U->Q[7] + Q.s[0]*U->Q[6];
    U->Q[6] = t1;
    U->Q[7] = t2;

    return;
}

void ft_apply_reflection(ft_reflection Q, ft_orthogonal_transformation * U) {
    double t, gamma = 2.0/(Q.w[0]*Q.w[0] + Q.w[1]*Q.w[1] + Q.w[2]*Q.w[2]);

    t = gamma*(Q.w[0]*U->Q[0] + Q.w[1]*U->Q[1] + Q.w[2]*U->Q[2]);
    U->Q[0] -= t*Q.w[0];
    U->Q[1] -= t*Q.w[1];
    U->Q[2] -= t*Q.w[2];

    t = gamma*(Q.w[0]*U->Q[3] + Q.w[1]*U->Q[4] + Q.w[2]*U->Q[5]);
    U->Q[3] -= t*Q.w[0];
    U->Q[4] -= t*Q.w[1];
    U->Q[5] -= t*Q.w[2];

    t = gamma*(Q.w[0]*U->Q[6] + Q.w[1]*U->Q[7] + Q.w[2]*U->Q[8]);
    U->Q[6] -= t*Q.w[0];
    U->Q[7] -= t*Q.w[1];
    U->Q[8] -= t*Q.w[2];

    return;
}

void ft_execute_sph_polar_rotation(double * A, const int N, const int M, double s, double c) {
    double sold = 0.0, cold = 1.0, tc = 2.0*c, t1, t2;
    for (int m = 1; m <= M/2; m++) {
        for (int i = 0; i < N-m; i++) {
            t1 = A[i+N*(2*m-1)];
            t2 = A[i+N*(2*m)];
            A[i+N*(2*m-1)] = c*t1 + s*t2;
            A[i+N*(2*m  )] = c*t2 - s*t1;
        }
        t1 = s;
        t2 = c;
        s = tc*s-sold;
        c = tc*c-cold;
        sold = t1;
        cold = t2;
    }
}

void ft_execute_sph_polar_reflection(double * A, const int N, const int M) {
    for (int i = 1; i < N; i += 2)
        A[i] = -A[i];
    for (int m = 1; m <= M/2; m++)
        for (int i = 1; i < N-m; i += 2) {
            A[i+N*(2*m-1)] = -A[i+N*(2*m-1)];
            A[i+N*(2*m)] = -A[i+N*(2*m)];
        }
}

static inline double Gy_index(int l, int i, int j) {
    double num, den;
	if (l+2 <= i && i <= 2*l && i+j == 2*l) {
        num = (j+1)*(j+2);
        den = (2*l+1)*(2*l+3);
        return sqrt(num/den)/2.0;
    }
	else if (2 <= i && i <= l && i+j == 2*l+2) {
        num = (i-1)*i;
        den = (2*l+1)*(2*l+3);
        return -sqrt(num/den)/2.0;
    }
	else if (0 <= i && i <= l-1 && i+j == 2*l) {
        num = (2*l+1-i)*(2*l+2-i);
        den = (2*l+1)*(2*l+3);
        return -sqrt(num/den)/2.0;
    }
	else if (l+3 <= i && i <= 2*l+2 && i+j == 2*l+2) {
        num = (2*l+1-j)*(2*l+2-j);
        den = (2*l+1)*(2*l+3);
        return sqrt(num/den)/2.0;
    }
	else if (i == l+1 && j == l-1) {
        num = 2*l*(l+1);
        den = (2*l+1)*(2*l+3);
        return sqrt(num/den)/2.0;
    }
	else if (i == l && j == l) {
        num = 2*(l+1)*(l+2);
        den = (2*l+1)*(2*l+3);
        return -sqrt(num/den)/2.0;
    }
    else
        return 0.0;
}

static inline double Gz_index(int l, int i, int j) {
	if (i == j+1) {
        double num = (j+1)*(2*l+1-j);
        double den = (2*l+1)*(2*l+3);
        return sqrt(num/den);
    }
    else
        return 0.0;
}

static inline double Gy_index_i_plus_j_eq_2l_no_den(int l, int i, int j) {
    double num;
	if (l+2 <= i && i <= 2*l) {
        num = (j+1)*(j+2);
        return sqrt(num)/2.0;
    }
	else if (0 <= i && i <= l-1) {
        num = (2*l+1-i)*(2*l+2-i);
        return -sqrt(num)/2.0;
    }
	else if (i == l+1 && j == l-1) {
        num = 2*l*(l+1);
        return sqrt(num)/2.0;
    }
	else if (i == l && j == l) {
        num = 2*(l+1)*(l+2);
        return -sqrt(num)/2.0;
    }
    else
        return 0.0;
}

static inline double Gy_index_i_plus_j_eq_2l_plus_2_no_den(int l, int i, int j) {
    double num;
 if (2 <= i && i <= l) {
        num = (i-1)*i;
        return -sqrt(num)/2.0;
    }
	else if (l+3 <= i && i <= 2*l+2) {
        num = (2*l+1-j)*(2*l+2-j);
        return sqrt(num)/2.0;
    }
    else
        return 0.0;
}

static inline double Gy_index_squared_i_plus_j_eq_2l_no_den(int l, int i, int j) {
    double num;
	if (l+2 <= i && i <= 2*l) {
        num = (j+1)*(j+2);
        return 0.25*num;
    }
	else if (0 <= i && i <= l-1) {
        num = (2*l+1-i)*(2*l+2-i);
        return 0.25*num;
    }
	else if (i == l+1 && j == l-1) {
        num = 2*l*(l+1);
        return 0.25*num;
    }
	else if (i == l && j == l) {
        num = 2*(l+1)*(l+2);
        return 0.25*num;
    }
    else
        return 0.0;
}
static inline double Gy_index_squared_i_plus_j_eq_2l_plus_2_no_den(int l, int i, int j) {
    double num;
 if (2 <= i && i <= l) {
        num = (i-1)*i;
        return 0.25*num;
    }
	else if (l+3 <= i && i <= 2*l+2) {
        num = (2*l+1-j)*(2*l+2-j);
        return 0.25*num;
    }
 else
   return 0.0;
}

static inline double Y_index(int l, int i, int j) {
    return Gy_index(l, 2*l-i, i)*Gy_index(l, 2*l-i, j) + Gy_index(l, 2*l-i+1, i)*Gy_index(l, 2*l-i+1, j) + Gy_index(l, 2*l-i+2, i)*Gy_index(l, 2*l-i+2, j);
}

static inline double Y_index_j_eq_i_no_den(int l, int i) {
    return Gy_index_squared_i_plus_j_eq_2l_no_den(l, 2*l-i, i)
//         + Gy_index(l, 2*l-i+1, i)*Gy_index(l, 2*l-i+1, i)  // this term is always zero
         + Gy_index_squared_i_plus_j_eq_2l_plus_2_no_den(l, 2*l-i+2, i);
}

static inline double Y_index_j_eq_i_plus_2_no_den(int l, int i) {
    return Gy_index_i_plus_j_eq_2l_no_den(l, 2*l-i, i)
          *Gy_index_i_plus_j_eq_2l_plus_2_no_den(l, 2*l-i, i+2);
//         + Gy_index(l, 2*l-i+1, i)*Gy_index(l, 2*l-i+1, i+2)  // this term is always zero
//         + Gy_index(l, 2*l-i+2, i)*Gy_index(l, 2*l-i+2, i+2); // this term is always zero
}

static inline double Z_index(int l, int i, int j) {
	if (i == j) {
        double num = (j+1)*(2*l+1-j);
        double den = (2*l+1)*(2*l+3);
        return num/den;
    }
    else
        return 0.0;
}
static inline double Z_index_no_den(int l, int i, int j) {
	if (i == j) {
        double num = (j+1)*(2*l+1-j);
        return num;
    }
    else
        return 0.0;
}

void ft_destroy_partial_sph_isometry_plan(ft_partial_sph_isometry_plan * F) {
    ft_destroy_symmetric_tridiagonal_symmetric_eigen(F->F11);
    ft_destroy_symmetric_tridiagonal_symmetric_eigen(F->F21);
    ft_destroy_symmetric_tridiagonal_symmetric_eigen(F->F12);
    ft_destroy_symmetric_tridiagonal_symmetric_eigen(F->F22);
    free(F);
}

void ft_destroy_sph_isometry_plan(ft_sph_isometry_plan * F) {
    for (int l = 2; l < F->n; l++)
        ft_destroy_partial_sph_isometry_plan(F->F[l-2]);
    free(F);
}

ft_partial_sph_isometry_plan * ft_plan_partial_sph_isometry(const int l) {
    int sign;

    int n11 = l/2;
    ft_symmetric_tridiagonal * Y11 = malloc(sizeof(ft_symmetric_tridiagonal));
    double * a11 = malloc(n11*sizeof(double));
    double * b11 = malloc((n11-1)*sizeof(double));

    for (int i = 0; i < n11; i++)
        a11[n11-1-i] = Y_index_j_eq_i_no_den(l, 2*i+1);
    for (int i = 0; i < n11-1; i++)
        b11[n11-2-i] = Y_index_j_eq_i_plus_2_no_den(l, 2*i+1);

    Y11->a = a11;
    Y11->b = b11;
    Y11->n = n11;

    double * lambda11 = malloc(n11*sizeof(double));
    for (int i = 0; i < n11; i++)
        lambda11[n11-1-i] = Z_index_no_den(l, 2*i+1, 2*i+1);

    sign = (l%4)/2 == 1 ? 1 : -1;

    ft_symmetric_tridiagonal_symmetric_eigen * F11 = ft_symmetric_tridiagonal_symmetric_eig(Y11, lambda11, sign);

    int n21 = (l+1)/2;
    ft_symmetric_tridiagonal * Y21 = malloc(sizeof(ft_symmetric_tridiagonal));
    double * a21 = malloc(n21*sizeof(double));
    double * b21 = malloc((n21-1)*sizeof(double));

    for (int i = 0; i < n21; i++)
        a21[n21-1-i] = Y_index_j_eq_i_no_den(l, 2*i);
    for (int i = 0; i < n21-1; i++)
        b21[n21-2-i] = Y_index_j_eq_i_plus_2_no_den(l, 2*i);

    Y21->a = a21;
    Y21->b = b21;
    Y21->n = n21;

    double * lambda21 = malloc(n21*sizeof(double));
    for (int i = 0; i < n21; i++)
        lambda21[i] = Z_index_no_den(l, l+1-l%2+2*i, l+1-l%2+2*i);

    sign = ((l+1)%4)/2 == 1 ? -1 : 1;

    ft_symmetric_tridiagonal_symmetric_eigen * F21 = ft_symmetric_tridiagonal_symmetric_eig(Y21, lambda21, sign);

    int n12 = (l+1)/2;
    ft_symmetric_tridiagonal * Y12 = malloc(sizeof(ft_symmetric_tridiagonal));
    double * a12 = malloc(n12*sizeof(double));
    double * b12 = malloc((n12-1)*sizeof(double));

    for (int i = 0; i < n12; i++)
        a12[i] = Y_index_j_eq_i_no_den(l, 2*i+l-l%2+1);
    for (int i = 0; i < n12-1; i++)
        b12[i] = Y_index_j_eq_i_plus_2_no_den(l, 2*i+l-l%2+1);

    Y12->a = a12;
    Y12->b = b12;
    Y12->n = n12;

    double * lambda12 = malloc(n12*sizeof(double));
    for (int i = 0; i < n12; i++)
        lambda12[n12-1-i] = Z_index_no_den(l, 2*i, 2*i);

    ft_symmetric_tridiagonal_symmetric_eigen * F12 = ft_symmetric_tridiagonal_symmetric_eig(Y12, lambda12, sign);

    int n22 = (l+2)/2;
    ft_symmetric_tridiagonal * Y22 = malloc(sizeof(ft_symmetric_tridiagonal));
    double * a22 = malloc(n22*sizeof(double));
    double * b22 = malloc((n22-1)*sizeof(double));

    for (int i = 0; i < n22; i++)
        a22[i] = Y_index_j_eq_i_no_den(l, 2*i+l+l%2);
    for (int i = 0; i < n22-1; i++)
        b22[i] = Y_index_j_eq_i_plus_2_no_den(l, 2*i+l+l%2);

    Y22->a = a22;
    Y22->b = b22;
    Y22->n = n22;

    double * lambda22 = malloc(n22*sizeof(double));
    for (int i = 0; i < n22; i++)
        lambda22[i] = Z_index_no_den(l, l+l%2+2*i, l+l%2+2*i);

    sign = (l%4)/2 == 1 ? -1 : 1;

    ft_symmetric_tridiagonal_symmetric_eigen * F22 = ft_symmetric_tridiagonal_symmetric_eig(Y22, lambda22, sign);

    free(lambda11);
    free(lambda21);
    free(lambda12);
    free(lambda22);
    ft_destroy_symmetric_tridiagonal(Y11);
    ft_destroy_symmetric_tridiagonal(Y21);
    ft_destroy_symmetric_tridiagonal(Y12);
    ft_destroy_symmetric_tridiagonal(Y22);

    ft_partial_sph_isometry_plan * F = malloc(sizeof(ft_partial_sph_isometry_plan));
    F->F11 = F11;
    F->F21 = F21;
    F->F12 = F12;
    F->F22 = F22;
    F->l = l;

    return F;
}

ft_sph_isometry_plan * ft_plan_sph_isometry(const int n) {
    ft_sph_isometry_plan * F = malloc(sizeof(ft_sph_isometry_plan));
    F->F = malloc((n-2)*sizeof(ft_partial_sph_isometry_plan *));
    for (int l = 2; l < n; l++)
        F->F[l-2] = ft_plan_partial_sph_isometry(l);
    F->n = n;
    return F;
}

void ft_execute_sph_yz_axis_exchange(ft_sph_isometry_plan * J, double * A, const int N, const int M) {
    if (J->n > 0) {
        double t = -A[1];
        A[1] = -A[N];
        A[N] = t;
    }
    #pragma omp parallel
    for (int l = 2 + FT_GET_THREAD_NUM(); l < J->n; l += FT_GET_NUM_THREADS()) {
        ft_partial_sph_isometry_plan * F = J->F[l-2];
        int stride = 4*N-2;
        int outshifta = N*(2*l-1)+N-l;
        int outshiftb = N*(2*l)+N-l;
        int inshift11 = l+N-1+(l%2)*(2*N-1);
        int inshift22 = l+(l%2)*(2*N-1);
        ft_semv(F->F11, A+inshift11, stride, A+outshifta);
        ft_semv(F->F22, A+inshift22, stride, A+outshiftb);
        for (int i = 0; i < F->F11->n; i++) {
            A[i*stride+inshift11] = A[i+outshifta];
            A[i+outshifta] = 0.0;
        }
        for (int i = 0; i < F->F22->n; i++) {
            A[i*stride+inshift22] = A[i+outshiftb];
            A[i+outshiftb] = 0.0;
        }
        int inshift21 = l-1+N+(1-(l%2))*(2*N-1);
        int inshift12 = l + (1-(l%2))*(2*N-1);
        ft_semv(F->F21, A+inshift21, stride, A+outshifta);
        ft_semv(F->F12, A+inshift12, stride, A+outshiftb);
        for (int i = 0; i < F->F21->n; i++) {
            A[i*stride+inshift21] = A[i+outshiftb];
            A[i+outshiftb] = 0.0;
            A[i*stride+inshift12] = A[i+outshifta];
            A[i+outshifta] = 0.0;
        }
    }
}

void ft_execute_sph_isometry(ft_sph_isometry_plan * J, ft_ZYZR Q, double * A, const int N, const int M) {
    if (Q.sign < 0)
        ft_execute_sph_polar_reflection(A, N, M);
    ft_execute_sph_polar_rotation(A, N, M, Q.s[2], Q.c[2]);
    ft_execute_sph_yz_axis_exchange(J, A, N, M);
    ft_execute_sph_polar_rotation(A, N, M, -Q.s[1], Q.c[1]);
    ft_execute_sph_yz_axis_exchange(J, A, N, M);
    ft_execute_sph_polar_rotation(A, N, M, Q.s[0], Q.c[0]);
}

void ft_execute_sph_rotation(ft_sph_isometry_plan * J, const double alpha, const double beta, const double gamma, double * A, const int N, const int M) {
    ft_ZYZR F = {{sin(alpha), sin(beta), sin(gamma)}, {cos(alpha), cos(beta), cos(gamma)}, 1};
    ft_execute_sph_isometry(J, F, A, N, M);
}

void ft_execute_sph_reflection(ft_sph_isometry_plan * J, ft_reflection W, double * A, const int N, const int M) {
    ft_orthogonal_transformation U = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    ft_apply_reflection(W, &U);
    ft_execute_sph_orthogonal_transformation(J, U, A, N, M);
}

void ft_execute_sph_orthogonal_transformation(ft_sph_isometry_plan * J, ft_orthogonal_transformation Q, double * A, const int N, const int M) {
    ft_execute_sph_isometry(J, ft_create_ZYZR(Q), A, N, M);
}
