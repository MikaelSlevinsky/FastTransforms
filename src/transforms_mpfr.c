void ft_mpfr_destroy_plan(mpfr_t * A, int n) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            mpfr_clear(A[i+j*n]);
    free(A);
}

void ft_mpfr_destroy_triangular_banded(ft_mpfr_triangular_banded * A) {
    for (int j = 0; j < A->n; j++)
        for (int i = 0; i < A->b+1; i++)
            mpfr_clear(A->data[i+j*(A->b+1)]);
    free(A->data);
    free(A);
}

ft_mpfr_triangular_banded * ft_mpfr_calloc_triangular_banded(const int n, const int b, mpfr_prec_t prec) {
    mpfr_t * data = malloc(n*(b+1)*sizeof(mpfr_t));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < b+1; i++) {
            mpfr_init2(data[i+j*(b+1)], prec);
            mpfr_set_zero(data[i+j*(b+1)], 1);
        }
    ft_mpfr_triangular_banded * A = malloc(sizeof(ft_mpfr_triangular_banded));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}

void ft_mpfr_get_triangular_banded_index(const ft_mpfr_triangular_banded * A, mpfr_t * v, const int i, const int j, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j-i && j-i <= b && i < n && j < n)
        mpfr_set(* v, A->data[i+(j+1)*b], rnd);
    else
        mpfr_set_zero(* v, 1);
    return;
}

void ft_mpfr_set_triangular_banded_index(const ft_mpfr_triangular_banded * A, const mpfr_t v, const int i, const int j, mpfr_rnd_t rnd) {
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j-i && j-i <= b && i < n && j < n)
        mpfr_set(A->data[i+(j+1)*b], v, rnd);
}

void ft_mpfr_triangular_banded_eigenvalues(ft_mpfr_triangular_banded * A, ft_mpfr_triangular_banded * B, mpfr_t * lambda, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1, t2;
    for (int j = 0; j < A->n; j++) {
        mpfr_init2(t1, prec);
        mpfr_init2(t2, prec);
        ft_mpfr_get_triangular_banded_index(A, &t1, j, j, prec, rnd);
        ft_mpfr_get_triangular_banded_index(B, &t2, j, j, prec, rnd);
        mpfr_div(lambda[j], t1, t2, rnd);
        mpfr_clear(t1);
        mpfr_clear(t2);
    }
}

// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void ft_mpfr_triangular_banded_eigenvectors(ft_mpfr_triangular_banded * A, ft_mpfr_triangular_banded * B, mpfr_t * V, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    mpfr_t t, t1, t2, t3, t4, lam;
    for (int j = 1; j < n; j++) {
        //lam = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
        mpfr_init2(t1, prec);
        mpfr_init2(t2, prec);
        ft_mpfr_get_triangular_banded_index(A, &t1, j, j, prec, rnd);
        ft_mpfr_get_triangular_banded_index(B, &t2, j, j, prec, rnd);
        mpfr_init2(lam, prec);
        mpfr_div(lam, t1, t2, rnd);
        mpfr_clear(t1);
        mpfr_clear(t2);
        for (int i = j-1; i >= 0; i--) {
            //t = 0;
            mpfr_init2(t, prec);
            mpfr_set_zero(t, 1);
            for (int k = i+1; k < MIN(i+b+1, n); k++) {
                //t += (lam*X(get_triangular_banded_index)(B, i, k) - X(get_triangular_banded_index)(A, i, k))*V[k+j*n];
                mpfr_init2(t3, prec);
                mpfr_set(t3, V[k+j*n], rnd);
                mpfr_init2(t4, prec);
                mpfr_init2(t1, prec);
                mpfr_init2(t2, prec);
                ft_mpfr_get_triangular_banded_index(A, &t1, i, k, prec, rnd);
                ft_mpfr_get_triangular_banded_index(B, &t2, i, k, prec, rnd);
                mpfr_fms(t4, lam, t2, t1, rnd);
                mpfr_fma(t, t4, t3, t, rnd);
                mpfr_clear(t1);
                mpfr_clear(t2);
                mpfr_clear(t3);
                mpfr_clear(t4);
            }
            //V[i+j*n] = -t/(lam*X(get_triangular_banded_index)(B, i, i) - X(get_triangular_banded_index)(A, i, i));
            mpfr_init2(t1, prec);
            mpfr_init2(t2, prec);
            ft_mpfr_get_triangular_banded_index(A, &t1, i, i, prec, rnd);
            ft_mpfr_get_triangular_banded_index(B, &t2, i, i, prec, rnd);
            mpfr_init2(t3, prec);
            mpfr_fms(t3, lam, t2, t1, rnd);
            mpfr_init2(t4, prec);
            mpfr_div(t4, t, t3, rnd);
            mpfr_neg(V[i+j*n], t4, rnd);
            mpfr_clear(t1);
            mpfr_clear(t2);
            mpfr_clear(t3);
            mpfr_clear(t4);
        }
    }
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_A_legendre_to_chebyshev(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v;
    mpfr_init2(v, prec);
    if (n > 1) {
        mpfr_set_d(v, 2.0, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(v, -i*(i-1.0), rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i-2, i, rnd);
        mpfr_set_d(v, i*(i+1.0), rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    return A;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_B_legendre_to_chebyshev(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * B = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v;
    mpfr_init2(v, prec);
    if (n > 0) {
        mpfr_set_d(v, 2.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 0, rnd);
    }
    if (n > 1) {
        mpfr_set_d(v, 1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(v, -1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i-2, i, rnd);
        mpfr_set_d(v, 1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    return B;
}

mpfr_t * ft_mpfr_plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_create_A_legendre_to_chebyshev(n, prec, rnd);
    ft_mpfr_triangular_banded * B = ft_mpfr_create_B_legendre_to_chebyshev(n, prec, rnd);
    mpfr_t * V = malloc(n*n*sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i+j*n], prec);
            mpfr_set_zero(V[i+j*n], 1);
        }
        mpfr_set_d(V[j+j*n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
    mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
    mpfr_t t, t1, sqrtpi, sqrtpi2;
    mpfr_init2(t, prec);
    mpfr_init2(t1, prec);
    mpfr_set_d(t, 1.0, rnd);
    mpfr_t half;
    mpfr_init2(half, prec);
    mpfr_set_d(half, 0.5, rnd);
    mpfr_init2(sqrtpi, prec);
    mpfr_gamma(sqrtpi, half, rnd);
    mpfr_t sqrthalf;
    mpfr_init2(sqrthalf, prec);
    mpfr_sqrt(sqrthalf, half, rnd);
    mpfr_init2(sqrtpi2, prec);
    mpfr_mul(sqrtpi2, sqrtpi, sqrthalf, rnd);

    if (n > 0) {
        //sclrow[0] = normcheb ? sqrtpi : 1;
        mpfr_init2(sclrow[0], prec);
        normcheb ? mpfr_set(sclrow[0], sqrtpi, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = normleg ? Y2(sqrt)(0.5) : 1;
        mpfr_init2(sclcol[0], prec);
        normleg ? mpfr_set(sclcol[0], sqrthalf, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    if (n > 1) {
        //sclrow[1] = normcheb ? sqrtpi2 : 1;
        mpfr_init2(sclrow[1], prec);
        normcheb ? mpfr_set(sclrow[1], sqrtpi2, rnd) : mpfr_set_d(sclrow[1], 1.0, rnd);
        //sclcol[1] = normleg ? Y2(sqrt)(1.5) : 1;
        mpfr_init2(sclcol[1], prec);
        mpfr_set_d(t1, 1.5, rnd);
        normleg ? mpfr_sqrt(sclcol[1], t1, rnd) : mpfr_set_d(sclcol[1], 1.0, rnd);
    }
    mpfr_t num, den, rat;
    mpfr_init2(num, prec);
    mpfr_init2(den, prec);
    mpfr_init2(rat, prec);
    for (int i = 2; i < n; i++) {
        //t *= (2*i-ONE(FLT2))/(2*i);
        mpfr_set_d(num, 2*i-1, rnd);
        mpfr_set_d(den, 2*i, rnd);
        mpfr_div(rat, num, den, rnd);
        mpfr_mul(t, rat, t, rnd);
        //sclrow[i] = normcheb ? sqrtpi2 : 1;
        mpfr_init2(sclrow[i], prec);
        normcheb ? mpfr_set(sclrow[i], sqrtpi2, rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = (normleg ? Y2(sqrt)(i+0.5) : 1)*t;
        mpfr_init2(sclcol[i], prec);
        mpfr_set_d(t1, i+0.5, rnd);
        normleg ? mpfr_sqrt(sclcol[i], t1, rnd) : mpfr_set_d(sclcol[i], 1.0, rnd);
        mpfr_mul(sclcol[i], t, sclcol[i], rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i+j*n], sclrow[i], V[i+j*n], rnd);
            mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t);
    mpfr_clear(t1);
    mpfr_clear(sqrtpi);
    mpfr_clear(sqrtpi2);
    mpfr_clear(half);
    mpfr_clear(sqrthalf);
    mpfr_clear(num);
    mpfr_clear(den);
    mpfr_clear(rat);
    return V;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_A_chebyshev_to_legendre(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, w, x;
    mpfr_init2(v, prec);
    mpfr_init2(w, prec);
    mpfr_init2(x, prec);
    if (n > 1) {
        mpfr_set_d(w, 1, rnd);
        mpfr_set_d(x, 3, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(w, -(i+1.0)*(i+1.0), rnd);
        mpfr_set_d(x, 2*i+1, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i-2, i, rnd);
        mpfr_set_d(w, 1.0*i*i, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(w);
    mpfr_clear(x);
    return A;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_B_chebyshev_to_legendre(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * B = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, w, x;
    mpfr_init2(v, prec);
    mpfr_init2(w, prec);
    mpfr_init2(x, prec);
    if (n > 0) {
        mpfr_set_d(v, 1, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 0, rnd);
    }
    if (n > 1) {
        mpfr_set_d(w, 1, rnd);
        mpfr_set_d(x, 3, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(w, -1, rnd);
        mpfr_set_d(x, 2*i+1, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i-2, i, rnd);
        mpfr_set_d(w, 1, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(w);
    mpfr_clear(x);
    return B;
}

mpfr_t * ft_mpfr_plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_create_A_chebyshev_to_legendre(n, prec, rnd);
    ft_mpfr_triangular_banded * B = ft_mpfr_create_B_chebyshev_to_legendre(n, prec, rnd);
    mpfr_t * V = malloc(n*n*sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i+j*n], prec);
            mpfr_set_zero(V[i+j*n], 1);
        }
        mpfr_set_d(V[j+j*n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
    mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
    mpfr_t t, t1, sqrtpi, sqrt_1_pi, sqrt_2_pi;
    mpfr_init2(t, prec);
    mpfr_init2(t1, prec);
    mpfr_set_d(t, 1.0, rnd);
    mpfr_t half;
    mpfr_init2(half, prec);
    mpfr_set_d(half, 0.5, rnd);
    mpfr_init2(sqrtpi, prec);
    mpfr_gamma(sqrtpi, half, rnd);
    mpfr_init2(sqrt_1_pi, prec);
    mpfr_div(sqrt_1_pi, t, sqrtpi, rnd);
    mpfr_t sqrt2;
    mpfr_init2(sqrt2, prec);
    mpfr_sqrt_ui(sqrt2, 2, rnd);
    mpfr_init2(sqrt_2_pi, prec);
    mpfr_mul(sqrt_2_pi, sqrt_1_pi, sqrt2, rnd);

    if (n > 0) {
        //sclrow[0] = normleg ? 1/Y2(sqrt)(0.5) : 1;
        mpfr_init2(sclrow[0], prec);
        normleg ? mpfr_set(sclrow[0], sqrt2, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = normcheb ? sqrt_1_pi : 1;
        mpfr_init2(sclcol[0], prec);
        normcheb ? mpfr_set(sclcol[0], sqrt_1_pi, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    if (n > 1) {
        //sclrow[1] = normleg ? 1/Y2(sqrt)(1.5) : 1;
        mpfr_init2(sclrow[1], prec);
        mpfr_set_d(t1, 1.5, rnd);
        normleg ? mpfr_rec_sqrt(sclrow[1], t1, rnd) : mpfr_set_d(sclrow[1], 1.0, rnd);
        //sclcol[1] = normcheb ? sqrt_2_pi : 1;
        mpfr_init2(sclcol[1], prec);
        normcheb ? mpfr_set(sclcol[1], sqrt_2_pi, rnd) : mpfr_set_d(sclcol[1], 1.0, rnd);
    }
    mpfr_t num, den, rat;
    mpfr_init2(num, prec);
    mpfr_init2(den, prec);
    mpfr_init2(rat, prec);
    for (int i = 2; i < n; i++) {
        //t *= (2*i)/(2*i-ONE(FLT2));
        mpfr_set_d(num, 2*i, rnd);
        mpfr_set_d(den, 2*i-1, rnd);
        mpfr_div(rat, num, den, rnd);
        mpfr_mul(t, rat, t, rnd);
        //sclrow[i] = normleg ? 1/Y2(sqrt)(i+0.5) : 1;
        mpfr_init2(sclrow[i], prec);
        mpfr_set_d(t1, i+0.5, rnd);
        normleg ? mpfr_rec_sqrt(sclrow[i], t1, rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = (normcheb ? sqrt_2_pi : 1)*t;
        mpfr_init2(sclcol[i], prec);
        normcheb ? mpfr_set(sclcol[i], sqrt_2_pi, rnd) : mpfr_set_d(sclcol[i], 1.0, rnd);
        mpfr_mul(sclcol[i], t, sclcol[i], rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i+j*n], sclrow[i], V[i+j*n], rnd);
            mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t);
    mpfr_clear(t1);
    mpfr_clear(sqrtpi);
    mpfr_clear(sqrt_1_pi);
    mpfr_clear(sqrt_2_pi);
    mpfr_clear(half);
    mpfr_clear(sqrt2);
    mpfr_clear(num);
    mpfr_clear(den);
    mpfr_clear(rat);
    return V;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_A_ultraspherical_to_ultraspherical(const int n, mpfr_srcptr lambda, mpfr_srcptr mu, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, t1, t2, t3, t4, t5, t6, t7, t8;
    mpfr_init2(v, prec);
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);
    mpfr_init2(t4, prec);
    mpfr_init2(t5, prec);
    mpfr_init2(t6, prec);
    mpfr_init2(t7, prec);
    mpfr_init2(t8, prec);
    if (n > 1) {
        // v = (1+2*lambda)*mu/(1+mu);
        mpfr_mul_d(t1, lambda, 2, rnd);
        mpfr_add_d(t2, t1, 1, rnd);
        mpfr_mul(t3, t2, mu, rnd);
        mpfr_add_d(t4, mu, 1, rnd);
        mpfr_div(v, t3, t4, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        // v = -(i+2*mu)*(i+2*(mu-lambda))*mu/(i+mu);
        mpfr_mul_d(t1, mu, 2, rnd);
        mpfr_add_d(t2, t1, i, rnd);
        mpfr_sub(t3, mu, lambda, rnd);
        mpfr_mul_d(t4, t3, 2, rnd);
        mpfr_add_d(t5, t4, i, rnd);
        mpfr_mul(t6, t2, t5, rnd);
        // t7 = mu/(i+mu);
        mpfr_add_d(t7, mu, i, rnd);
        mpfr_div(t7, mu, t7, rnd);
        mpfr_mul(t8, t6, t7, rnd);
        mpfr_neg(v, t8, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i-2, i, rnd);
        // v = i*(i+2*lambda)*mu/(i+mu);
        mpfr_mul_d(t1, lambda, 2, rnd);
        mpfr_add_d(t2, t1, i, rnd);
        mpfr_mul_d(t3, t2, i, rnd);
        mpfr_mul(v, t3, t7, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    mpfr_clear(t4);
    mpfr_clear(t5);
    mpfr_clear(t6);
    mpfr_clear(t7);
    mpfr_clear(t8);
    return A;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_B_ultraspherical_to_ultraspherical(const int n, mpfr_srcptr mu, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * B = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, t1, t2;
    mpfr_init2(v, prec);
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    if (n > 0) {
        mpfr_set_d(v, 1, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 0, rnd);
    }
    if (n > 1) {
        mpfr_add_d(t1, mu, 1, rnd);
        mpfr_div(v, mu, t1, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_add_d(t1, mu, i, rnd);
        mpfr_div(v, mu, t1, rnd);
        mpfr_neg(t2, v, rnd);
        ft_mpfr_set_triangular_banded_index(B, t2, i-2, i, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(t1);
    mpfr_clear(t2);
    return B;
}

mpfr_t * ft_mpfr_plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, mpfr_srcptr lambda, mpfr_srcptr mu, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_create_A_ultraspherical_to_ultraspherical(n, lambda, mu, prec, rnd);
    ft_mpfr_triangular_banded * B = ft_mpfr_create_B_ultraspherical_to_ultraspherical(n, mu, prec, rnd);
    mpfr_t * V = malloc(n*n*sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i+j*n], prec);
            mpfr_set_zero(V[i+j*n], 1);
        }
        mpfr_set_d(V[j+j*n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
    mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
    mpfr_t t1, t2, t3, t4, t5, t6, t7;
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);
    mpfr_init2(t4, prec);
    mpfr_init2(t5, prec);
    mpfr_init2(t6, prec);
    mpfr_init2(t7, prec);

    if (n > 0) {
        //sclrow[0] = norm2 ? Y2(sqrt)(Y2(tgamma)(0.5)*Y2(tgamma)(mu2+0.5)/Y2(tgamma)(mu2+1)) : 1;
        mpfr_set_d(t1, 0.5, rnd);
        mpfr_gamma(t2, t1, rnd);
        mpfr_add_d(t3, mu, 0.5, rnd);
        mpfr_gamma(t4, t3, rnd);
        mpfr_add_d(t5, mu, 1.0, rnd);
        mpfr_gamma(t6, t5, rnd);
        mpfr_mul(t7, t2, t4, rnd);
        mpfr_div(t7, t7, t6, rnd);
        mpfr_sqrt(t7, t7, rnd);
        mpfr_init2(sclrow[0], prec);
        norm2 ? mpfr_set(sclrow[0], t7, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = norm1 ? Y2(sqrt)(Y2(tgamma)(lambda2+1)/(Y2(tgamma)(0.5)*Y2(tgamma)(lambda2+0.5))) : 1;
        mpfr_add_d(t3, lambda, 1.0, rnd);
        mpfr_gamma(t4, t3, rnd);
        mpfr_add_d(t5, lambda, 0.5, rnd);
        mpfr_gamma(t6, t5, rnd);
        mpfr_mul(t7, t2, t6, rnd);
        mpfr_div(t7, t4, t7, rnd);
        mpfr_sqrt(t7, t7, rnd);
        mpfr_init2(sclcol[0], prec);
        norm1 ? mpfr_set(sclcol[0], t7, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    for (int i = 1; i < n; i++) {
        //sclrow[i] = norm2 ? Y2(sqrt)((i-1+mu2)/i*(i-1+2*mu2)/(i+mu2))*sclrow[i-1] : 1;
        mpfr_add_d(t1, mu, i-1, rnd);
        mpfr_div_d(t2, t1, i, rnd);
        mpfr_add(t3, t1, mu, rnd);
        mpfr_add_d(t4, t1, 1, rnd);
        mpfr_div(t5, t3, t4, rnd);
        mpfr_mul(t6, t2, t5, rnd);
        mpfr_sqrt(t6, t6, rnd);
        mpfr_mul(t7, t6, sclrow[i-1], rnd);
        mpfr_init2(sclrow[i], prec);
        norm2 ? mpfr_set(sclrow[i], t7, rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = norm1 ? Y2(sqrt)(i/(i-1+lambda2)*(i+lambda2)/(i-1+2*lambda2))*(i-1+lambda2)/(i-1+mu2)*sclcol[i-1] : (i-1+lambda2)/(i-1+mu2)*sclcol[i-1];
        mpfr_add_d(t1, lambda, i-1, rnd);
        mpfr_d_div(t2, i, t1, rnd);
        mpfr_add_d(t3, t1, 1, rnd);
        mpfr_add(t4, t1, lambda, rnd);
        mpfr_div(t5, t3, t4, rnd);
        mpfr_mul(t6, t2, t5, rnd);
        mpfr_sqrt(t7, t6, rnd);
        // t1 = (i-1+lambda)
        mpfr_add_d(t2, mu, i-1, rnd);
        mpfr_div(t3, t1, t2, rnd);
        mpfr_mul(t4, t3, sclcol[i-1], rnd);
        mpfr_init2(sclcol[i], prec);
        norm1 ? mpfr_mul(sclcol[i], t7, t4, rnd) : mpfr_set(sclcol[i], t4, rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i+j*n], sclrow[i], V[i+j*n], rnd);
            mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    mpfr_clear(t4);
    mpfr_clear(t5);
    mpfr_clear(t6);
    mpfr_clear(t7);
    return V;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_A_jacobi_to_jacobi(const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr gamma, mpfr_srcptr delta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    mpfr_init2(v, prec);
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);
    mpfr_init2(t4, prec);
    mpfr_init2(t5, prec);
    mpfr_init2(t6, prec);
    mpfr_init2(t7, prec);
    mpfr_init2(t8, prec);
    mpfr_init2(t9, prec);
    mpfr_init2(t10, prec);
    mpfr_init2(t11, prec);
    mpfr_init2(t12, prec);
    mpfr_init2(t13, prec);
    mpfr_init2(t14, prec);
    mpfr_init2(t15, prec);
    if (n > 1) {
        mpfr_add(t1, alpha, beta, rnd);
        mpfr_sub(t2, alpha, beta, rnd);
        mpfr_add(t3, gamma, delta, rnd);
        mpfr_sub(t4, gamma, delta, rnd);
        mpfr_sub(t1, t3, t1, rnd);
        mpfr_div_d(t1, t1, 2, rnd);
        mpfr_sub(t2, t4, t2, rnd);
        mpfr_div_d(t2, t2, 2, rnd);
        mpfr_add_d(t3, t3, 2, rnd);
        mpfr_add_d(t5, t3, 2, rnd);
        // v = (gamma-delta)*(gamma+delta+2)/(gamma+delta+4)*(1+(gamma-alpha+delta-beta)/2) - (gamma+delta+2)*(gamma-alpha+beta-delta)/2;
        mpfr_add_d(v, t1, 1, rnd);
        mpfr_mul(v, v, t3, rnd);
        mpfr_div(v, v, t5, rnd);
        mpfr_mul(v, v, t4, rnd);
        mpfr_mul(t2, t2, t3, rnd);
        mpfr_sub(v, v, t2, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 0, 1, rnd);
        // v = (alpha+beta+2)*(gamma+delta+2)/(gamma+delta+4);
        mpfr_add(v, alpha, beta, rnd);
        mpfr_add_d(v, v, 2, rnd);
        mpfr_mul(v, v, t3, rnd);
        mpfr_div(v, v, t5, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_add_d(t1, gamma, i, rnd);
        mpfr_add_d(t2, delta, i, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_add_d(t4, t3, 1, rnd);
        mpfr_add_d(t5, t4, 1, rnd);
        mpfr_add(t6, alpha, beta, rnd);
        mpfr_sub(t7, alpha, beta, rnd);
        mpfr_add(t8, gamma, delta, rnd);
        mpfr_sub(t9, gamma, delta, rnd);
        mpfr_add_d(t10, t8, i+1, rnd);
        mpfr_add_d(t11, t10, 1, rnd);
        mpfr_add_d(t12, t6, i+1, rnd);
        mpfr_sub(t13, t8, t6, rnd);
        mpfr_add_d(t14, t13, i, rnd);
        mpfr_div_d(t13, t13, 2, rnd);
        mpfr_add_d(t8, t8, 2, rnd);
        mpfr_mul(t8, t8, t13, rnd);
        mpfr_sub(t15, t9, t7, rnd);
        mpfr_div_d(t15, t15, 2, rnd);
        mpfr_mul(t15, t10, t15, rnd);
        // v = -(i+gamma+delta+1)*(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1)*(i+gamma-alpha+delta-beta)
        mpfr_mul(v, t10, t1, rnd);
        mpfr_div(v, v, t3, rnd);
        mpfr_mul(v, v, t2, rnd);
        mpfr_div(v, v, t4, rnd);
        mpfr_mul(v, v, t14, rnd);
        mpfr_neg(v, v, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i-2, i, rnd);
        // v = (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2)*(i*(i+gamma+delta+1)+(gamma+delta+2)*(gamma-alpha+delta-beta)/2) - (i+gamma+delta+1)*(gamma-alpha+beta-delta)/2;
        mpfr_mul_d(v, t10, i, rnd);
        mpfr_add(v, v, t8, rnd);
        mpfr_mul(v, v, t9, rnd);
        mpfr_div(v, v, t3, rnd);
        mpfr_mul(v, v, t10, rnd);
        mpfr_div(v, v, t5, rnd);
        mpfr_sub(v, v, t15, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i-1, i, rnd);
        // v = i*(i+alpha+beta+1)*(i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2);
        mpfr_mul_d(v, t12, i, rnd);
        mpfr_mul(v, v, t10, rnd);
        mpfr_div(v, v, t4, rnd);
        mpfr_mul(v, v, t11, rnd);
        mpfr_div(v, v, t5, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    mpfr_clear(t4);
    mpfr_clear(t5);
    mpfr_clear(t6);
    mpfr_clear(t7);
    mpfr_clear(t8);
    mpfr_clear(t9);
    mpfr_clear(t10);
    mpfr_clear(t11);
    mpfr_clear(t12);
    mpfr_clear(t13);
    mpfr_clear(t14);
    mpfr_clear(t15);
    return A;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_B_jacobi_to_jacobi(const int n, mpfr_srcptr gamma, mpfr_srcptr delta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * B = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, t1, t2, t3, t4, t5, t6;
    mpfr_init2(v, prec);
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);
    mpfr_init2(t4, prec);
    mpfr_init2(t5, prec);
    mpfr_init2(t6, prec);
    if (n > 0) {
        // v = 1;
        mpfr_set_d(v, 1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 0, rnd);
    }
    if (n > 1) {
        // v = (gamma-delta)/(gamma+delta+4);
        mpfr_sub(t1, gamma, delta, rnd);
        mpfr_add(t2, gamma, delta, rnd);
        mpfr_add_d(t2, t2, 4, rnd);
        mpfr_div(v, t1, t2, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 1, rnd);
        // v = (gamma+delta+2)/(gamma+delta+4);
        mpfr_add(t1, gamma, delta, rnd);
        mpfr_add_d(t1, t1, 2, rnd);
        mpfr_div(v, t1, t2, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        // v = -(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1);
        mpfr_add_d(t1, gamma, i, rnd);
        mpfr_add_d(t2, delta, i, rnd);
        mpfr_add(t3, t1, t2, rnd); // Preserve t3.
        mpfr_add_d(t4, t3, 1, rnd);
        mpfr_div(t5, t1, t3, rnd);
        mpfr_div(t6, t2, t4, rnd);
        mpfr_mul(v, t5, t6, rnd);
        mpfr_neg(v, v, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i-2, i, rnd);
        // v = (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2);
        mpfr_sub(t1, gamma, delta, rnd);
        mpfr_sub_d(t2, t3, i-1, rnd); // Preserve t2.
        mpfr_add_d(t4, t3, 2, rnd);
        mpfr_div(t5, t1, t3, rnd);
        mpfr_div(t6, t2, t4, rnd);
        mpfr_mul(v, t5, t6, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i-1, i, rnd);
        // v = (i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2);
        mpfr_add_d(t1, t3, 1, rnd);
        mpfr_add_d(t4, t2, 1, rnd);
        mpfr_add_d(t3, t3, 2, rnd);
        mpfr_div(t5, t2, t1, rnd);
        mpfr_div(t6, t4, t3, rnd);
        mpfr_mul(v, t5, t6, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    mpfr_clear(t4);
    mpfr_clear(t5);
    mpfr_clear(t6);
    return B;
}

mpfr_t * ft_mpfr_plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr gamma, mpfr_srcptr delta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_create_A_jacobi_to_jacobi(n, alpha, beta, gamma, delta, prec, rnd);
    ft_mpfr_triangular_banded * B = ft_mpfr_create_B_jacobi_to_jacobi(n, gamma, delta, prec, rnd);
    mpfr_t * V = malloc(n*n*sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i+j*n], prec);
            mpfr_set_zero(V[i+j*n], 1);
        }
        mpfr_set_d(V[j+j*n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
    mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
    mpfr_t t1, t2, t3, t4;
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);
    mpfr_init2(t4, prec);

    if (n > 0) {
        //sclrow[0] = norm2 ? Y2(sqrt)(Y2(pow)(2, gamma2+delta2+1)*Y2(tgamma)(gamma2+1)*Y2(tgamma)(delta2+1)/Y2(tgamma)(gamma2+delta2+2)) : 1;
        mpfr_add_d(t1, gamma, 1, rnd);
        mpfr_add_d(t2, delta, 1, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_sub_d(t4, t3, 1, rnd);
        mpfr_gamma(t1, t1, rnd);
        mpfr_gamma(t2, t2, rnd);
        mpfr_gamma(t3, t3, rnd);
        mpfr_ui_pow(t4, 2, t4, rnd);
        mpfr_div(t3, t3, t1, rnd);
        mpfr_div(t3, t3, t2, rnd);
        mpfr_div(t3, t3, t4, rnd);
        mpfr_rec_sqrt(t3, t3, rnd);
        mpfr_init2(sclrow[0], prec);
        norm2 ? mpfr_set(sclrow[0], t3, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = norm1 ? Y2(sqrt)(Y2(tgamma)(alpha2+beta2+2)/(Y2(pow)(2, alpha2+beta2+1)*Y2(tgamma)(alpha2+1)*Y2(tgamma)(beta2+1))) : 1;
        mpfr_add_d(t1, alpha, 1, rnd);
        mpfr_add_d(t2, beta, 1, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_sub_d(t4, t3, 1, rnd);
        mpfr_gamma(t1, t1, rnd);
        mpfr_gamma(t2, t2, rnd);
        mpfr_gamma(t3, t3, rnd);
        mpfr_ui_pow(t4, 2, t4, rnd);
        mpfr_div(t3, t3, t1, rnd);
        mpfr_div(t3, t3, t2, rnd);
        mpfr_div(t3, t3, t4, rnd);
        mpfr_sqrt(t3, t3, rnd);
        mpfr_init2(sclcol[0], prec);
        norm1 ? mpfr_set(sclcol[0], t3, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    if (n > 1) {
        //sclrow[1] = norm2 ? Y2(sqrt)((gamma2+1)*(delta2+1)/(gamma2+delta2+3))*sclrow[0] : 1;
        mpfr_add_d(t1, gamma, 1, rnd);
        mpfr_add_d(t2, delta, 1, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_add_d(t3, t3, 1, rnd);
        mpfr_mul(t1, t1, t2, rnd);
        mpfr_div(t3, t1, t3, rnd);
        mpfr_sqrt(t3, t3, rnd);
        mpfr_init2(sclrow[1], prec);
        norm2 ? mpfr_mul(sclrow[1], t3, sclrow[0], rnd) : mpfr_set_d(sclrow[1], 1.0, rnd);
        //sclcol[1] = norm1 ? Y2(sqrt)((alpha2+beta2+3)/(alpha2+1)/(beta2+1))*(alpha2+beta2+2)/(gamma2+delta2+2)*sclcol[0] : (alpha2+beta2+2)/(gamma2+delta2+2);
        mpfr_add_d(t1, alpha, 1, rnd);
        mpfr_add_d(t2, beta, 1, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_add_d(t3, t3, 1, rnd);
        mpfr_mul(t1, t1, t2, rnd);
        mpfr_div(t3, t3, t1, rnd);
        mpfr_sqrt(t3, t3, rnd);
        mpfr_add(t1, alpha, beta, rnd);
        mpfr_add_d(t1, t1, 2, rnd);
        mpfr_add(t2, gamma, delta, rnd);
        mpfr_add_d(t2, t2, 2, rnd);
        mpfr_div(t4, t1, t2, rnd);
        mpfr_mul(t4, t4, sclcol[0], rnd);
        mpfr_init2(sclcol[1], prec);
        norm1 ? mpfr_mul(sclcol[1], t3, t4, rnd) : mpfr_set(sclcol[1], t4, rnd);
    }
    for (int i = 2; i < n; i++) {
        //sclrow[i] = norm2 ? Y2(sqrt)((i+gamma2)/i*(i+delta2)/(i+gamma2+delta2)*(2*i+gamma2+delta2-1)/(2*i+gamma2+delta2+1))*sclrow[i-1] : 1;
        mpfr_add_d(t1, gamma, i, rnd);
        mpfr_add_d(t2, delta, i, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_sub_d(t3, t3, 1, rnd);
        mpfr_add_d(t4, t3, 2, rnd);
        mpfr_div_d(t1, t1, i, rnd);
        mpfr_mul(t1, t1, t2, rnd);
        mpfr_add(t2, t2, gamma, rnd);
        mpfr_div(t1, t1, t2, rnd);
        mpfr_div(t3, t3, t4, rnd);
        mpfr_mul(t3, t1, t3, rnd);
        mpfr_sqrt(t3, t3, rnd);
        mpfr_init2(sclrow[i], prec);
        norm2 ? mpfr_mul(sclrow[i], t3, sclrow[i-1], rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = norm1 ? Y2(sqrt)(i/(i+alpha2)*(i+alpha2+beta2)/(i+beta2)*(2*i+alpha2+beta2+1)/(2*i+alpha2+beta2-1))*(2*i+alpha2+beta2-1)/(i+alpha2+beta2)*(2*i+alpha2+beta2)/(2*i+gamma2+delta2-1)*(i+gamma2+delta2)/(2*i+gamma2+delta2)*sclcol[i-1] : (2*i+alpha2+beta2-1)/(i+alpha2+beta2)*(2*i+alpha2+beta2)/(2*i+gamma2+delta2-1)*(i+gamma2+delta2)/(2*i+gamma2+delta2)*sclcol[i-1];
        mpfr_add_d(t1, alpha, i, rnd);
        mpfr_add_d(t2, beta, i, rnd);
        mpfr_add(t3, t1, t2, rnd);
        mpfr_sub_d(t3, t3, 1, rnd);
        mpfr_add_d(t4, t3, 2, rnd);
        mpfr_div_d(t1, t1, i, rnd);
        mpfr_mul(t1, t1, t2, rnd);
        mpfr_add(t2, t2, alpha, rnd);
        mpfr_div(t1, t1, t2, rnd);
        mpfr_div(t3, t3, t4, rnd);
        mpfr_mul(t3, t1, t3, rnd);
        mpfr_rec_sqrt(t3, t3, rnd);
        mpfr_add(t2, alpha, beta, rnd);
        mpfr_add_d(t2, t2, i, rnd);
        mpfr_add_d(t1, t2, i, rnd);
        mpfr_div(t4, t1, t2, rnd);
        mpfr_sub_d(t1, t1, 1, rnd);
        mpfr_mul(t4, t1, t4, rnd);
        mpfr_add(t1, gamma, delta, rnd);
        mpfr_add_d(t1, t1, i, rnd);
        mpfr_add_d(t2, t1, i, rnd);
        mpfr_mul(t4, t1, t4, rnd);
        mpfr_div(t4, t4, t2, rnd);
        mpfr_sub_d(t2, t2, 1, rnd);
        mpfr_div(t4, t4, t2, rnd);
        mpfr_mul(t4, t4, sclcol[i-1], rnd);
        mpfr_init2(sclcol[i], prec);
        norm1 ? mpfr_mul(sclcol[i], t3, t4, rnd) : mpfr_set(sclcol[i], t4, rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i+j*n], sclrow[i], V[i+j*n], rnd);
            mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    mpfr_clear(t4);
    return V;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_A_laguerre_to_laguerre(const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_calloc_triangular_banded(n, 1, prec);
    mpfr_t v;
    mpfr_init2(v, prec);
    for (int i = 0; i < n; i++) {
        mpfr_sub(v, alpha, beta, rnd);
        mpfr_sub_d(v, v, i, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i-1, i, rnd);
        mpfr_set_d(v, i, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    return A;
}

static inline ft_mpfr_triangular_banded * ft_mpfr_create_B_laguerre_to_laguerre(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * B = ft_mpfr_calloc_triangular_banded(n, 1, prec);
    mpfr_t v;
    mpfr_init2(v, prec);
    for (int i = 0; i < n; i++) {
        mpfr_set_d(v, -1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i-1, i, rnd);
        mpfr_set_d(v, 1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    mpfr_clear(v);
    return B;
}

mpfr_t * ft_mpfr_plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded * A = ft_mpfr_create_A_laguerre_to_laguerre(n, alpha, beta, prec, rnd);
    ft_mpfr_triangular_banded * B = ft_mpfr_create_B_laguerre_to_laguerre(n, prec, rnd);
    mpfr_t * V = malloc(n*n*sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i+j*n], prec);
            mpfr_set_zero(V[i+j*n], 1);
        }
        mpfr_set_d(V[j+j*n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
    mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
    mpfr_t t1, t2, t3;
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);

    if (n > 0) {
        //sclrow[0] = norm2 ? Y2(sqrt)(Y2(tgamma)(beta2+1)) : 1;
        mpfr_add_d(t1, beta, 1.0, rnd);
        mpfr_gamma(t2, t1, rnd);
        mpfr_sqrt(t3, t2, rnd);
        mpfr_init2(sclrow[0], prec);
        norm2 ? mpfr_set(sclrow[0], t3, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = norm1 ? 1/Y2(sqrt)(Y2(tgamma)(alpha2+1)) : 1;
        mpfr_add_d(t1, alpha, 1.0, rnd);
        mpfr_gamma(t2, t1, rnd);
        mpfr_rec_sqrt(t3, t2, rnd);
        mpfr_init2(sclcol[0], prec);
        norm1 ? mpfr_set(sclcol[0], t3, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    for (int i = 1; i < n; i++) {
        //sclrow[i] = norm2 ? Y2(sqrt)((i+beta2)/i)*sclrow[i-1] : 1;
        mpfr_add_d(t1, beta, i, rnd);
        mpfr_div_d(t2, t1, i, rnd);
        mpfr_sqrt(t3, t2, rnd);
        mpfr_init2(sclrow[i], prec);
        norm2 ? mpfr_mul(sclrow[i], t3, sclrow[i-1], rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = norm1 ? Y2(sqrt)(i/(i+alpha2))*sclcol[i-1] : 1;
        mpfr_add_d(t1, alpha, i, rnd);
        mpfr_d_div(t2, i, t1, rnd);
        mpfr_sqrt(t3, t2, rnd);
        mpfr_init2(sclcol[i], prec);
        norm1 ? mpfr_mul(sclcol[i], t3, sclcol[i-1], rnd) : mpfr_set_d(sclcol[i], 1.0, rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i+j*n], sclrow[i], V[i+j*n], rnd);
            mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    return V;
}

mpfr_t * ft_mpfr_plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1;
    mpfr_init2(t1, prec);
    mpfr_sub_d(t1, lambda, 0.5, rnd);
    mpfr_t * V = ft_mpfr_plan_jacobi_to_jacobi(normjac, normultra, n, alpha, beta, t1, t1, prec, rnd);
    if (normultra == 0) {
        mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
        if (n > 0) {
            mpfr_init2(sclrow[0], prec);
            mpfr_set_d(sclrow[0], 1.0, rnd);
        }
        mpfr_t t2;
        mpfr_init2(t2, prec);
        mpfr_mul_d(t2, lambda, 2, rnd);
        mpfr_sub_d(t2, t2, 1, rnd);
        for (int i = 1; i < n; i++) {
            //sclrow[i] = (lambda+i-0.5)/(2*lambda+i-1)*sclrow[i-1];
            mpfr_add_d(t1, t1, 1, rnd);
            mpfr_add_d(t2, t2, 1, rnd);
            mpfr_init2(sclrow[i], prec);
            mpfr_div(sclrow[i], t1, t2, rnd);
            mpfr_mul(sclrow[i], sclrow[i], sclrow[i-1], rnd);
        }
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                //V[i+j*n] *= sclrow[i];
                mpfr_mul(V[i+j*n], V[i+j*n], sclrow[i], rnd);
        for (int i = 0; i < n; i++)
            mpfr_clear(sclrow[i]);
        free(sclrow);
        mpfr_clear(t2);
    }
    mpfr_clear(t1);
    return V;
}

mpfr_t * ft_mpfr_plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, mpfr_srcptr lambda, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1;
    mpfr_init2(t1, prec);
    mpfr_sub_d(t1, lambda, 0.5, rnd);
    mpfr_t * V = ft_mpfr_plan_jacobi_to_jacobi(normultra, normjac, n, t1, t1, alpha, beta, prec, rnd);
    if (normultra == 0) {
        mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
        if (n > 0) {
            mpfr_init2(sclcol[0], prec);
            mpfr_set_d(sclcol[0], 1.0, rnd);
        }
        mpfr_t t2;
        mpfr_init2(t2, prec);
        mpfr_mul_d(t2, lambda, 2, rnd);
        mpfr_sub_d(t2, t2, 1, rnd);
        for (int i = 1; i < n; i++) {
            //sclcol[i] = (2*lambda+i-1)/(lambda+i-0.5)*sclcol[i-1];
            mpfr_add_d(t1, t1, 1, rnd);
            mpfr_add_d(t2, t2, 1, rnd);
            mpfr_init2(sclcol[i], prec);
            mpfr_div(sclcol[i], t2, t1, rnd);
            mpfr_mul(sclcol[i], sclcol[i], sclcol[i-1], rnd);
        }
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                //V[i+j*n] *= sclcol[j];
                mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        for (int i = 0; i < n; i++)
            mpfr_clear(sclcol[i]);
        free(sclcol);
        mpfr_clear(t2);
    }
    mpfr_clear(t1);
    return V;
}

mpfr_t * ft_mpfr_plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1;
    mpfr_init2(t1, prec);
    mpfr_set_d(t1, -0.5, rnd);
    mpfr_t * V = ft_mpfr_plan_jacobi_to_jacobi(normjac, 1, n, alpha, beta, t1, t1, prec, rnd);
    if (normcheb == 0) {
        mpfr_t sqrt_1_pi, sqrt_2_pi;
        mpfr_neg(t1, t1, rnd);
        mpfr_init2(sqrt_1_pi, prec);
        mpfr_gamma(sqrt_1_pi, t1, rnd);
        mpfr_d_div(sqrt_1_pi, 1.0, sqrt_1_pi, rnd);
        mpfr_init2(sqrt_2_pi, prec);
        mpfr_sqrt(sqrt_2_pi, t1, rnd);
        mpfr_div(sqrt_2_pi, sqrt_1_pi, sqrt_2_pi, rnd);
        mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
        for (int i = 0; i < n; i++) {
            mpfr_init2(sclrow[i], prec);
            i ? mpfr_set(sclrow[i], sqrt_2_pi, rnd) : mpfr_set(sclrow[i], sqrt_1_pi, rnd);
        }
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                mpfr_mul(V[i+j*n], V[i+j*n], sclrow[i], rnd);
        for (int i = 0; i < n; i++)
            mpfr_clear(sclrow[i]);
        free(sclrow);
        mpfr_clear(sqrt_1_pi);
        mpfr_clear(sqrt_2_pi);
    }
    mpfr_clear(t1);
    return V;
}

mpfr_t * ft_mpfr_plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1;
    mpfr_init2(t1, prec);
    mpfr_set_d(t1, -0.5, rnd);
    mpfr_t * V = ft_mpfr_plan_jacobi_to_jacobi(1, normjac, n, t1, t1, alpha, beta, prec, rnd);
    if (normcheb == 0) {
        mpfr_t sqrtpi, sqrtpi2;
        mpfr_neg(t1, t1, rnd);
        mpfr_init2(sqrtpi, prec);
        mpfr_gamma(sqrtpi, t1, rnd);
        mpfr_init2(sqrtpi2, prec);
        mpfr_sqrt(sqrtpi2, t1, rnd);
        mpfr_mul(sqrtpi2, sqrtpi, sqrtpi2, rnd);
        mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
        for (int i = 0; i < n; i++) {
            mpfr_init2(sclcol[i], prec);
            i ? mpfr_set(sclcol[i], sqrtpi2, rnd) : mpfr_set(sclcol[i], sqrtpi, rnd);
        }
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        for (int i = 0; i < n; i++)
            mpfr_clear(sclcol[i]);
        free(sclcol);
        mpfr_clear(sqrtpi);
        mpfr_clear(sqrtpi2);
    }
    mpfr_clear(t1);
    return V;
}

mpfr_t * ft_mpfr_plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1;
    mpfr_init2(t1, prec);
    mpfr_set_d(t1, -0.5, rnd);
    mpfr_t * V = ft_mpfr_plan_ultraspherical_to_jacobi(normultra, 1, n, lambda, t1, t1, prec, rnd);
    if (normcheb == 0) {
        mpfr_t sqrt_1_pi, sqrt_2_pi;
        mpfr_neg(t1, t1, rnd);
        mpfr_init2(sqrt_1_pi, prec);
        mpfr_gamma(sqrt_1_pi, t1, rnd);
        mpfr_d_div(sqrt_1_pi, 1.0, sqrt_1_pi, rnd);
        mpfr_init2(sqrt_2_pi, prec);
        mpfr_sqrt(sqrt_2_pi, t1, rnd);
        mpfr_div(sqrt_2_pi, sqrt_1_pi, sqrt_2_pi, rnd);
        mpfr_t * sclrow = malloc(n*sizeof(mpfr_t));
        for (int i = 0; i < n; i++) {
            mpfr_init2(sclrow[i], prec);
            i ? mpfr_set(sclrow[i], sqrt_2_pi, rnd) : mpfr_set(sclrow[i], sqrt_1_pi, rnd);
        }
        for (int j = 0; j < n; j++)
            for (int i = j; i >= 0; i -= 2)
                mpfr_mul(V[i+j*n], V[i+j*n], sclrow[i], rnd);
        for (int i = 0; i < n; i++)
            mpfr_clear(sclrow[i]);
        free(sclrow);
        mpfr_clear(sqrt_1_pi);
        mpfr_clear(sqrt_2_pi);
    }
    mpfr_clear(t1);
    return V;
}

mpfr_t * ft_mpfr_plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1;
    mpfr_init2(t1, prec);
    mpfr_set_d(t1, -0.5, rnd);
    mpfr_t * V = ft_mpfr_plan_jacobi_to_ultraspherical(1, normultra, n, t1, t1, lambda, prec, rnd);
    if (normcheb == 0) {
        mpfr_t sqrtpi, sqrtpi2;
        mpfr_neg(t1, t1, rnd);
        mpfr_init2(sqrtpi, prec);
        mpfr_gamma(sqrtpi, t1, rnd);
        mpfr_init2(sqrtpi2, prec);
        mpfr_sqrt(sqrtpi2, t1, rnd);
        mpfr_mul(sqrtpi2, sqrtpi, sqrtpi2, rnd);
        mpfr_t * sclcol = malloc(n*sizeof(mpfr_t));
        for (int i = 0; i < n; i++) {
            mpfr_init2(sclcol[i], prec);
            i ? mpfr_set(sclcol[i], sqrtpi2, rnd) : mpfr_set(sclcol[i], sqrtpi, rnd);
        }
        for (int j = 0; j < n; j++)
            for (int i = j; i >= 0; i -= 2)
                mpfr_mul(V[i+j*n], V[i+j*n], sclcol[j], rnd);
        for (int i = 0; i < n; i++)
            mpfr_clear(sclcol[i]);
        free(sclcol);
        mpfr_clear(sqrtpi);
        mpfr_clear(sqrtpi2);
    }
    mpfr_clear(t1);
    return V;
}
