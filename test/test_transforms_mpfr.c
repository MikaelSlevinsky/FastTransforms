mpfr_t * ft_mpfr_init_Id(int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t * A = malloc(n*n*sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(A[i+j*n], prec);
            mpfr_set_zero(A[i+j*n], 1);
        }
        mpfr_set_d(A[j+j*n], 1.0, rnd);
    }
    return A;
}

void ft_mpfr_norm_1arg(mpfr_t * ret, mpfr_t * A, int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_set_d(* ret, 0.0, rnd);
    for (int i = 0; i < n; i++)
        mpfr_fma(* ret, A[i], A[i], * ret, rnd);
    mpfr_sqrt(* ret, * ret, rnd);
}

void ft_mpfr_norm_2arg(mpfr_t * ret, mpfr_t * A, mpfr_t * B, int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t;
    mpfr_set_d(* ret, 0.0, rnd);
    mpfr_init2(t, prec);
    for (int i = 0; i < n; i++) {
        mpfr_sub(t, A[i], B[i], rnd);
        mpfr_fma(* ret, t, t, * ret, rnd);
    }
    mpfr_sqrt(* ret, * ret, rnd);
    mpfr_clear(t);
}

void ft_mpfr_checktest(mpfr_t * err, double cst, int * checksum, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1, t2, one, oneplus, eps;
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(one, prec);
    mpfr_init2(oneplus, prec);
    mpfr_init2(eps, prec);
    mpfr_set_d(one, 1.0, rnd);
    mpfr_set_d(oneplus, 1.0, rnd);
    mpfr_nextabove(oneplus);
    mpfr_sub(eps, oneplus, one, rnd);
    mpfr_abs(t1, * err, rnd);
    mpfr_mul_d(t2, eps, cst, rnd);
    if (mpfr_cmp(t1, t2) < 0) printf(GREEN("✓")"\n");
    else {printf(RED("×")"\n"); (*checksum)++;}
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(one);
    mpfr_clear(oneplus);
    mpfr_clear(eps);
}

void test_transforms_mpfr(int * checksum, int N, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t err, t1, t2, t3;
    mpfr_init2(err, prec);
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_t * Id, * A, * B;

    printf("\nTesting the accuracy of Chebyshev--Legendre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        mpfr_set_d(err, 0.0, rnd);
        Id = ft_mpfr_init_Id(n, prec, rnd);
        B = ft_mpfr_init_Id(n, prec, rnd);
        for (int normleg = 0; normleg <= 1; normleg++) {
            for (int normcheb = 0; normcheb <= 1; normcheb++) {
                A = ft_mpfr_plan_legendre_to_chebyshev(normleg, normcheb, n, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                A = ft_mpfr_plan_chebyshev_to_legendre(normcheb, normleg, n, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                ft_mpfr_norm_2arg(&t1, B, Id, n*n, prec, rnd);
                ft_mpfr_norm_1arg(&t2, Id, n*n, prec, rnd);
                mpfr_div(t1, t1, t2, rnd);
                mpfr_add(err, err, t1, rnd);
            }
        }
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, mpfr_get_d(err, rnd));
        ft_mpfr_checktest(&err, n, checksum, prec, rnd);
        ft_mpfr_destroy_plan(Id, n);
        ft_mpfr_destroy_plan(B, n);
    }

    mpfr_t lambda, mu;
    mpfr_init2(lambda, prec);
    mpfr_init2(mu, prec);

    printf("\nTesting the accuracy of ultraspherical--ultraspherical transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        Id = ft_mpfr_init_Id(n, prec, rnd);
        for (int cases = 0; cases < 4; cases++) {
            mpfr_set_d(err, 0.0, rnd);
            B = ft_mpfr_init_Id(n, prec, rnd);
            switch (cases) {
                case 0:
                    mpfr_set_d(lambda, -0.125, rnd);
                    mpfr_set_d(mu, 0.125, rnd);
                    break;
                case 1:
                    mpfr_set_d(lambda, 1.5, rnd);
                    mpfr_set_d(mu, 1.0, rnd);
                    break;
                case 2:
                    mpfr_set_d(lambda, 0.25, rnd);
                    mpfr_set_d(mu, 1.25, rnd);
                    break;
                case 3:
                    mpfr_set_d(lambda, 0.5, rnd);
                    mpfr_set_d(mu, 2.5, rnd);
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = ft_mpfr_plan_ultraspherical_to_ultraspherical(norm1, norm2, n, lambda, mu, prec, rnd);
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    A = ft_mpfr_plan_ultraspherical_to_ultraspherical(norm2, norm1, n, mu, lambda, prec, rnd);
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    ft_mpfr_norm_2arg(&t1, B, Id, n*n, prec, rnd);
                    ft_mpfr_norm_1arg(&t2, Id, n*n, prec, rnd);
                    mpfr_div(t1, t1, t2, rnd);
                    mpfr_add(err, err, t1, rnd);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, mpfr_get_d(lambda, rnd), mpfr_get_d(mu, rnd), mpfr_get_d(err, rnd));
            ft_mpfr_checktest(&err, 4*pow(n, 2*fabs(mu-lambda)), checksum, prec, rnd);
            ft_mpfr_destroy_plan(B, n);
        }
        ft_mpfr_destroy_plan(Id, n);
    }

    mpfr_t alpha, beta, gamma, delta;
    mpfr_init2(alpha, prec);
    mpfr_init2(beta, prec);
    mpfr_init2(gamma, prec);
    mpfr_init2(delta, prec);

    printf("\nTesting the accuracy of Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        Id = ft_mpfr_init_Id(n, prec, rnd);
        for (int cases = 0; cases < 8; cases++) {
            mpfr_set_d(err, 0.0, rnd);
            B = ft_mpfr_init_Id(n, prec, rnd);
            switch (cases) {
                case 0:
                    mpfr_set_d(alpha, 0.0, rnd);
                    mpfr_set_d(beta, -0.5, rnd);
                    mpfr_set_d(gamma, -0.5, rnd);
                    mpfr_set_d(delta, -0.5, rnd);
                    break;
                case 1:
                    mpfr_set_d(alpha, 0.1, rnd);
                    mpfr_set_d(beta, 0.2, rnd);
                    mpfr_set_d(gamma, 0.3, rnd);
                    mpfr_set_d(delta, 0.4, rnd);
                    break;
                case 2:
                    mpfr_set_d(alpha, 1.0, rnd);
                    mpfr_set_d(beta, 0.5, rnd);
                    mpfr_set_d(gamma, 0.5, rnd);
                    mpfr_set_d(delta, 0.25, rnd);
                    break;
                case 3:
                    mpfr_set_d(alpha, -0.25, rnd);
                    mpfr_set_d(beta, -0.75, rnd);
                    mpfr_set_d(gamma, 0.25, rnd);
                    mpfr_set_d(delta, 0.75, rnd);
                    break;
                case 4:
                    mpfr_set_d(alpha, 0.0, rnd);
                    mpfr_set_d(beta, 1.0, rnd);
                    mpfr_set_d(gamma, -0.5, rnd);
                    mpfr_set_d(delta, 0.5, rnd);
                    break;
                case 5:
                    mpfr_set_d(alpha, 0.0, rnd);
                    mpfr_set_d(beta, -0.5, rnd);
                    mpfr_set_d(gamma, -0.5, rnd);
                    mpfr_set_d(delta, -0.25, rnd);
                    break;
                case 6:
                    mpfr_set_d(alpha, -0.5, rnd);
                    mpfr_set_d(beta, 0.5, rnd);
                    mpfr_set_d(gamma, -0.5, rnd);
                    mpfr_set_d(delta, 0.0, rnd);
                    break;
                case 7:
                    mpfr_set_d(alpha, 0.5, rnd);
                    mpfr_set_d(beta, -0.5, rnd);
                    mpfr_set_d(gamma, -0.5, rnd);
                    mpfr_set_d(delta, 0.0, rnd);
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = ft_mpfr_plan_jacobi_to_jacobi(norm1, norm2, n, alpha, beta, gamma, delta, prec, rnd);
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    A = ft_mpfr_plan_jacobi_to_jacobi(norm2, norm1, n, gamma, delta, alpha, beta, prec, rnd);
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    ft_mpfr_norm_2arg(&t1, B, Id, n*n, prec, rnd);
                    ft_mpfr_norm_1arg(&t2, Id, n*n, prec, rnd);
                    mpfr_div(t1, t1, t2, rnd);
                    mpfr_add(err, err, t1, rnd);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, mpfr_get_d(alpha, rnd), mpfr_get_d(beta, rnd), mpfr_get_d(gamma, rnd), mpfr_get_d(delta, rnd), mpfr_get_d(err, rnd));
            ft_mpfr_checktest(&err, 32*pow(n, 2*(MAX(fabs(gamma-alpha), fabs(delta-beta)))), checksum, prec, rnd);
            ft_mpfr_destroy_plan(B, n);
        }
        ft_mpfr_destroy_plan(Id, n);
    }

    printf("\nTesting the accuracy of Laguerre--Laguerre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        Id = ft_mpfr_init_Id(n, prec, rnd);
        for (int cases = 0; cases < 4; cases++) {
            mpfr_set_d(err, 0.0, rnd);
            B = ft_mpfr_init_Id(n, prec, rnd);
            switch (cases) {
                case 0:
                    mpfr_set_d(alpha, -0.125, rnd);
                    mpfr_set_d(beta, 0.125, rnd);
                    break;
                case 1:
                    mpfr_set_d(alpha, 1.5, rnd);
                    mpfr_set_d(beta, 1.0, rnd);
                    break;
                case 2:
                    mpfr_set_d(alpha, 0.25, rnd);
                    mpfr_set_d(beta, 1.25, rnd);
                    break;
                case 3:
                    mpfr_set_d(alpha, 0.5, rnd);
                    mpfr_set_d(beta, 2.0, rnd);
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = ft_mpfr_plan_laguerre_to_laguerre(norm1, norm2, n, alpha, beta, prec, rnd);
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    A = ft_mpfr_plan_laguerre_to_laguerre(norm2, norm1, n, beta, alpha, prec, rnd);
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    ft_mpfr_norm_2arg(&t1, B, Id, n*n, prec, rnd);
                    ft_mpfr_norm_1arg(&t2, Id, n*n, prec, rnd);
                    mpfr_div(t1, t1, t2, rnd);
                    mpfr_add(err, err, t1, rnd);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, mpfr_get_d(alpha, rnd), mpfr_get_d(beta, rnd), mpfr_get_d(err, rnd));
            ft_mpfr_checktest(&err, 4*pow(n, 2*fabs(alpha-beta)), checksum, prec, rnd);
            ft_mpfr_destroy_plan(B, n);
        }
        ft_mpfr_destroy_plan(Id, n);
    }

    printf("\nTesting the accuracy of interrelated transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        mpfr_set_d(err, 0.0, rnd);
        Id = ft_mpfr_init_Id(n, prec, rnd);
        B = ft_mpfr_init_Id(n, prec, rnd);
        for (int norm1 = 0; norm1 <= 1; norm1++) {
            for (int norm2 = 0; norm2 <= 1; norm2++) {
                A = ft_mpfr_plan_chebyshev_to_legendre(norm1, norm2, n, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                mpfr_set_d(lambda, 0.5, rnd);
                A = ft_mpfr_plan_ultraspherical_to_chebyshev(norm2, norm1, n, lambda, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                mpfr_set_d(lambda, 1.0, rnd);
                A = ft_mpfr_plan_chebyshev_to_ultraspherical(norm1, norm2, n, lambda, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                mpfr_set_d(alpha, 0.0, rnd);
                mpfr_set_d(beta, 0.0, rnd);
                A = ft_mpfr_plan_ultraspherical_to_jacobi(norm2, norm1, n, lambda, alpha, beta, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                A = ft_mpfr_plan_jacobi_to_chebyshev(norm1, norm2, n, alpha, beta, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                mpfr_set_d(alpha, 0.25, rnd);
                mpfr_set_d(beta, -0.25, rnd);
                A = ft_mpfr_plan_chebyshev_to_jacobi(norm2, norm1, n, alpha, beta, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                mpfr_set_d(lambda, 0.5, rnd);
                A = ft_mpfr_plan_jacobi_to_ultraspherical(norm1, norm2, n, alpha, beta, lambda, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                A = ft_mpfr_plan_legendre_to_chebyshev(norm2, norm1, n, prec, rnd);
                ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                ft_mpfr_destroy_plan(A, n);
                ft_mpfr_norm_2arg(&t1, B, Id, n*n, prec, rnd);
                ft_mpfr_norm_1arg(&t2, Id, n*n, prec, rnd);
                mpfr_div(t1, t1, t2, rnd);
                mpfr_add(err, err, t1, rnd);
            }
        }
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, mpfr_get_d(err, rnd));
        ft_mpfr_checktest(&err, n*n, checksum, prec, rnd);
        ft_mpfr_destroy_plan(Id, n);
        ft_mpfr_destroy_plan(B, n);
    }
    mpfr_clear(err);
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(alpha);
    mpfr_clear(beta);
    mpfr_clear(gamma);
    mpfr_clear(delta);
    mpfr_clear(lambda);
    mpfr_clear(mu);
}
