void Y(test_transforms)(int * checksum, int N) {
    FLT err;
    FLT * Id, * B, * x;
    X(tb_eigen_FMM) * A;
    X(btb_eigen_FMM) * C;
    X(banded) * M;

    printf("\nTesting the accuracy of Chebyshev--Legendre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        Id = calloc(n*n, sizeof(FLT));
        B = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            B[i+i*n] = Id[i+i*n] = 1;
        for (int normleg = 0; normleg <= 1; normleg++) {
            for (int normcheb = 0; normcheb <= 1; normcheb++) {
                A = X(plan_legendre_to_chebyshev)(normleg, normcheb, n);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_chebyshev_to_legendre)(normcheb, normleg, n);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                err += X(norm_2arg)(B, Id, n*n)/X(norm_1arg)(Id, n*n);
            }
        }
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, (double) err);
        X(checktest)(err, 2*Y(sqrt)(n), checksum);
        free(Id);
        free(B);
    }

    FLT lambda, mu;

    printf("\nTesting the accuracy of ultraspherical--ultraspherical transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        Id = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            Id[i+i*n] = 1;
        B = malloc(n*n*sizeof(FLT));
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++)
                    B[i+j*n] = 0;
                B[j+j*n] = 1;
            }
            switch (cases) {
                case 0:
                    lambda = -0.125;
                    mu = 0.125;
                    break;
                case 1:
                    lambda = 1.5;
                    mu = 1.0;
                    break;
                case 2:
                    lambda = 0.25;
                    mu = 1.25;
                    break;
                case 3:
                    lambda = 0.5;
                    mu = 2.5;
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = X(plan_ultraspherical_to_ultraspherical)(norm1, norm2, n, lambda, mu);
                    X(bfmm)('N', A, B, n, n);
                    X(destroy_tb_eigen_FMM)(A);
                    A = X(plan_ultraspherical_to_ultraspherical)(norm2, norm1, n, mu, lambda);
                    X(bfmm)('N', A, B, n, n);
                    X(destroy_tb_eigen_FMM)(A);
                    err += X(norm_2arg)(B, Id, n*n)/X(norm_1arg)(Id, n*n);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, (double) lambda, (double) mu, (double) err);
            X(checktest)(err, 4*Y(pow)(n, Y(fabs)(mu-lambda)), checksum);
        }
        free(Id);
        free(B);
    }

    FLT alpha, beta, gamma, delta;

    printf("\nTesting the accuracy of Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        Id = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            Id[i+i*n] = 1;
        B = malloc(n*n*sizeof(FLT));
        for (int cases = 0; cases < 8; cases++) {
            err = 0;
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++)
                    B[i+j*n] = 0;
                B[j+j*n] = 1;
            }
            switch (cases) {
                case 0:
                    alpha = 0.0;
                    beta = -0.5;
                    gamma = -0.5;
                    delta = -0.5;
                    break;
                case 1:
                    alpha = 0.1;
                    beta = 0.2;
                    gamma = 0.3;
                    delta = 0.4;
                    break;
                case 2:
                    alpha = 1.0;
                    beta = 0.5;
                    gamma = 0.5;
                    delta = 0.25;
                    break;
                case 3:
                    alpha = -0.25;
                    beta = -0.75;
                    gamma = 0.25;
                    delta = 0.75;
                    break;
                case 4:
                    alpha = 0.0;
                    beta = 1.0;
                    gamma = -0.5;
                    delta = 0.5;
                    break;
                case 5:
                    alpha = 0.0;
                    beta = -0.5;
                    gamma = -0.5;
                    delta = -0.25;
                    break;
                case 6:
                    alpha = -0.5;
                    beta = 0.5;
                    gamma = -0.5;
                    delta = 0.0;
                    break;
                case 7:
                    alpha = 0.5;
                    beta = -0.5;
                    gamma = -0.5;
                    delta = 0.0;
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = X(plan_jacobi_to_jacobi)(norm1, norm2, n, alpha, beta, gamma, delta);
                    X(bfmm)('N', A, B, n, n);
                    X(destroy_tb_eigen_FMM)(A);
                    A = X(plan_jacobi_to_jacobi)(norm2, norm1, n, gamma, delta, alpha, beta);
                    X(bfmm)('N', A, B, n, n);
                    X(destroy_tb_eigen_FMM)(A);
                    err += X(norm_2arg)(B, Id, n*n)/X(norm_1arg)(Id, n*n);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, (double) alpha, (double) beta, (double) gamma, (double) delta, (double) err);
            X(checktest)(err, 32*Y(pow)(n, MAX(Y(fabs)(gamma-alpha), Y(fabs)(delta-beta))), checksum);
        }
        free(Id);
        free(B);
    }

    printf("\nTesting the accuracy of Laguerre--Laguerre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        Id = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            Id[i+i*n] = 1;
        B = malloc(n*n*sizeof(FLT));
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++)
                    B[i+j*n] = 0;
                B[j+j*n] = 1;
            }
            switch (cases) {
                case 0:
                    alpha = -0.125;
                    beta = 0.125;
                    break;
                case 1:
                    alpha = 1.5;
                    beta = 1.0;
                    break;
                case 2:
                    alpha = 0.25;
                    beta = 1.25;
                    break;
                case 3:
                    alpha = 0.5;
                    beta = 2.0;
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = X(plan_laguerre_to_laguerre)(norm1, norm2, n, alpha, beta);
                    X(bfmm)('N', A, B, n, n);
                    X(destroy_tb_eigen_FMM)(A);
                    A = X(plan_laguerre_to_laguerre)(norm2, norm1, n, beta, alpha);
                    X(bfmm)('N', A, B, n, n);
                    X(destroy_tb_eigen_FMM)(A);
                    err += X(norm_2arg)(B, Id, n*n)/X(norm_1arg)(Id, n*n);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, (double) alpha, (double) beta, (double) err);
            X(checktest)(err, 4*Y(pow)(n, Y(fabs)(alpha-beta)), checksum);
        }
        free(Id);
        free(B);
    }

    printf("\nTesting the accuracy of interrelated classical transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        Id = calloc(n*n, sizeof(FLT));
        B = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            B[i+i*n] = Id[i+i*n] = 1;
        for (int norm1 = 0; norm1 <= 1; norm1++) {
            for (int norm2 = 0; norm2 <= 1; norm2++) {
                A = X(plan_chebyshev_to_legendre)(norm1, norm2, n);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_ultraspherical_to_chebyshev)(norm2, norm1, n, 0.5);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_chebyshev_to_ultraspherical)(norm1, norm2, n, 1.0);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_ultraspherical_to_jacobi)(norm2, norm1, n, 1.0, 0.0, 0.0);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_jacobi_to_chebyshev)(norm1, norm2, n, 0.0, 0.0);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_chebyshev_to_jacobi)(norm2, norm1, n, 0.25, -0.25);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_jacobi_to_ultraspherical)(norm1, norm2, n, 0.25, -0.25, 0.5);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                A = X(plan_legendre_to_chebyshev)(norm2, norm1, n);
                X(bfmm)('N', A, B, n, n);
                X(destroy_tb_eigen_FMM)(A);
                err += X(norm_2arg)(B, Id, n*n)/X(norm_1arg)(Id, n*n);
            }
        }
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, (double) err);
        X(checktest)(err, n, checksum);
        free(Id);
        free(B);
    }

    printf("\nTesting methods for associated Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N>>1; n *= 2) {
        B = malloc(n*n*sizeof(FLT));
        x = malloc(n*sizeof(FLT));
        for (int c = 1; c < 5; c++) {
            for (int cases = 0; cases < 7; cases++) {
                err = 0;
                switch (cases) {
                    case 0:
                        alpha = 0.0;
                        beta = -0.25;
                        gamma = -0.25;
                        delta = -0.5;
                        break;
                    case 1:
                        alpha = 0.1;
                        beta = 0.2;
                        gamma = 0.3;
                        delta = 0.4;
                        break;
                    case 2:
                        alpha = 1.0;
                        beta = 0.5;
                        gamma = 0.5;
                        delta = 0.25;
                        break;
                    case 3:
                        alpha = -0.25;
                        beta = 0.25;
                        gamma = 0.25;
                        delta = 0.75;
                        break;
                    case 4:
                        alpha = 0.0;
                        beta = 0.625;
                        gamma = -0.25;
                        delta = 0.5;
                        break;
                    case 5:
                        alpha = 0.5;
                        beta = -0.25;
                        gamma = 0.0;
                        delta = 0.0;
                        break;
                    case 6:
                        alpha = 0.0;
                        beta = 1.1920929e-7;
                        gamma = 0.0;
                        delta = 0.0;
                        break;
                }
                for (int norm1 = 0; norm1 <= 1; norm1++) {
                    for (int norm2 = 0; norm2 <= 1; norm2++) {
                        C = X(plan_associated_jacobi_to_jacobi)(norm1, norm2, n, c, alpha, beta, gamma, delta);
                        M = X(create_jacobi_multiplication)(norm2, n, n, gamma, delta);
                        for (int j = 0; j < n; j++) {
                            for (int i = 0; i < n; i++)
                                B[i+j*n] = 0;
                            B[j+j*n] = 1;
                        }
                        X(bbbfmm)('N', '2', '1', C, B, n, n);
                        for (int nu = 1; nu < n-1; nu++) {
                            for (int i = 0; i < n; i++)
                                x[i] = X(rec_B_jacobi)(norm1, nu+c, alpha, beta)*B[i+nu*n] - X(rec_C_jacobi)(norm1, nu+c, alpha, beta)*B[i+(nu-1)*n];
                            X(gbmv)(X(rec_A_jacobi)(norm1, nu+c, alpha, beta), M, B+nu*n, 1, x);
                            err += X(norm_2arg)(B+(nu+1)*n, x, n)/X(norm_1arg)(x, n);
                        }
                        X(destroy_btb_eigen_FMM)(C);
                        X(destroy_banded)(M);
                    }
                }
                printf("(n, c) = (%4i, %i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, c, (double) alpha, (double) beta, (double) gamma, (double) delta, (double) err);
                X(checktest)(err, ((FLT) n)*n*n, checksum);
            }
        }
        free(B);
        free(x);
    }
}

static inline FLT X(norm_2arg_banded_tridiagonal)(X(banded) * A, X(symmetric_tridiagonal) * T) {
    int n = A->n;
    FLT s = X(get_banded_index)(A, n-1, n-1) - T->a[n-1], t;
    FLT ret = s*s;
    for (int j = 0; j < n-1; j++) {
        s = X(get_banded_index)(A, j, j) - T->a[j];
        t = X(get_banded_index)(A, j, j+1) - T->b[j];
        ret += s*s+2*t*t;
    }
    return Y(sqrt)(ret);
}


void Y(test_modified_transforms)(int * checksum, int N) {
    FLT alpha, beta, err, u[5], v[5];
    FLT * DP, * IDP, * Id;
    X(modified_plan) * P;
    X(banded) * XP, * XQ;
    X(symmetric_tridiagonal) * JP, * JQ;

    printf("\nTesting the accuracy of modified Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 16; n < N; n *= 2) {
        Id = calloc(n*n, sizeof(FLT));
        for (int j = 0; j < n; j++)
            Id[j+j*n] = 1;
        alpha = 0;
        beta = 0;
        // u(x) = (1-x)^2*(1+x), v(x) = 1
        u[0] = 0.9428090415820636;
        u[1] = -0.32659863237109055;
        u[2] = -0.42163702135578396;
        u[3] = 0.2138089935299396;
        v[0] = 1.4142135623730951;
        P = X(plan_modified_jacobi_to_jacobi)(n, alpha, beta, 4, u, 1, v, 0);
        DP = calloc(n*n, sizeof(FLT));
        IDP = calloc(n*n, sizeof(FLT));
        for (int j = 0; j < n; j++) {
            IDP[j+j*n] = DP[j+j*n] = 1;
            X(mpmv)('N', P, DP+j*n);
            X(mpsv)('N', P, IDP+j*n);
        }
        XP = X(create_jacobi_multiplication)(1, n, n, alpha, beta);
        JP = X(convert_banded_to_symmetric_tridiagonal)(XP);
        JQ = X(execute_jacobi_similarity)(P, JP);
        XQ = X(create_jacobi_multiplication)(1, n-1, n-1, alpha+2, beta+1);
        err = X(norm_2arg_banded_tridiagonal)(XQ, JQ)/X(norm_1arg)(XQ->data, 3*(n-1));
        printf("Jacobi matrix from trivial rational weight \t n = %3i |%20.2e ", n-1, err);
        X(checktest)(err, Y(pow)(n+1, 2), checksum);
        X(destroy_symmetric_tridiagonal)(JQ);
        X(destroy_modified_plan)(P);
        P = X(plan_modified_jacobi_to_jacobi)(n, alpha, beta, 4, u, 0, NULL, 0);
        X(mpsm)('N', P, DP, n, n);
        X(mpmm)('N', P, IDP, n, n);
        err = X(norm_2arg)(DP, Id, n*n)/X(norm_1arg)(Id, n*n) + X(norm_2arg)(IDP, Id, n*n)/X(norm_1arg)(Id, n*n);
        printf("Polynomial vs. trivial rational weight \t\t n = %3i |%20.2e ", n, (double) err);
        X(checktest)(err, Y(pow)(n+1, 3), checksum);
        JQ = X(execute_jacobi_similarity)(P, JP);
        err = X(norm_2arg_banded_tridiagonal)(XQ, JQ)/X(norm_1arg)(XQ->data, 3*(n-1));
        printf("Jacobi matrix from polynomial modification \t n = %3i |%20.2e ", n-1, err);
        X(checktest)(err, Y(pow)(n+1, 2), checksum);
        X(destroy_banded)(XQ);
        X(destroy_symmetric_tridiagonal)(JP);
        X(destroy_symmetric_tridiagonal)(JQ);
        X(mpsm)('N', P, IDP, n, n);
        X(destroy_modified_plan)(P);
        alpha = 2;
        beta = 1;
        // u(x) = 1, v(x) = (2-x)*(2+x)
        u[0] = 1.1547005383792517;
        v[0] = 4.387862045841156;
        v[1] = 0.1319657758147716;
        v[2] = -0.20865621238292037;
        P = X(plan_modified_jacobi_to_jacobi)(n, alpha, beta, 1, u, 3, v, 0);
        X(mpsm)('N', P, IDP, n, n); // Should be the equivalent of the raising to (1-x)^2*(1+x)/(2-x)/(2+x)
        X(destroy_modified_plan)(P);
        alpha = 0;
        beta = 0;
        // u(x) = -(1-x)^2*(1+x), v(x) = -(2-x)*(2+x)
        u[0] = -0.9428090415820636;
        u[1] = 0.32659863237109055;
        u[2] = 0.42163702135578396;
        u[3] = -0.2138089935299396;
        v[0] = -5.185449728701348;
        v[1] = 0;
        v[2] = 0.42163702135578374;
        P = X(plan_modified_jacobi_to_jacobi)(n, alpha, beta, 4, u, 3, v, 0);
        X(mpmm)('N', P, DP, n, n);
        X(destroy_modified_plan)(P);
        X(trmm)('N', n, IDP, n, DP, n, n);
        err = X(norm_2arg)(DP, Id, n*n)/X(norm_1arg)(Id, n*n);
        printf("Rational vs. raised recip. polynomial weight \t n = %3i |%20.2e ", n, (double) err);
        X(checktest)(err, Y(pow)(n+1, 3), checksum);
        free(Id);
        free(DP);
        free(IDP);
    }

    printf("\nTesting the accuracy of modified Laguerre--Laguerre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 16; n < N; n *= 2) {
        Id = calloc(n*n, sizeof(FLT));
        for (int j = 0; j < n; j++)
            Id[j+j*n] = 1;
        alpha = 0;
        // u(x) = x^2, v(x) = 1
        u[0] = 2;
        u[1] = -4;
        u[2] = 2;
        v[0] = 1;
        P = X(plan_modified_laguerre_to_laguerre)(n, alpha, 3, u, 1, v, 0);
        DP = calloc(n*n, sizeof(FLT));
        IDP = calloc(n*n, sizeof(FLT));
        for (int j = 0; j < n; j++) {
            IDP[j+j*n] = DP[j+j*n] = 1;
            X(mpmv)('N', P, DP+j*n);
            X(mpsv)('N', P, IDP+j*n);
        }
        XP = X(create_laguerre_multiplication)(1, n, n, alpha);
        JP = X(convert_banded_to_symmetric_tridiagonal)(XP);
        JQ = X(execute_jacobi_similarity)(P, JP);
        XQ = X(create_laguerre_multiplication)(1, n-1, n-1, alpha+2);
        err = X(norm_2arg_banded_tridiagonal)(XQ, JQ)/X(norm_1arg)(XQ->data, 3*(n-1));
        printf("Jacobi matrix from trivial rational weight \t n = %3i |%20.2e ", n-1, err);
        X(checktest)(err, Y(pow)(n+1, 2), checksum);
        X(destroy_symmetric_tridiagonal)(JQ);
        X(destroy_modified_plan)(P);
        P = X(plan_modified_laguerre_to_laguerre)(n, alpha, 3, u, 0, NULL, 0);
        X(mpsm)('N', P, DP, n, n);
        X(mpmm)('N', P, IDP, n, n);
        err = X(norm_2arg)(DP, Id, n*n)/X(norm_1arg)(Id, n*n) + X(norm_2arg)(IDP, Id, n*n)/X(norm_1arg)(Id, n*n);
        printf("Polynomial vs. trivial rational weight \t\t n = %3i |%20.2e ", n, (double) err);
        X(checktest)(err, Y(pow)(n+1, 3), checksum);
        JQ = X(execute_jacobi_similarity)(P, JP);
        err = X(norm_2arg_banded_tridiagonal)(XQ, JQ)/X(norm_1arg)(XQ->data, 3*(n-1));
        printf("Jacobi matrix from polynomial modification \t n = %3i |%20.2e ", n-1, err);
        X(checktest)(err, Y(pow)(n+1, 2), checksum);
        X(destroy_banded)(XQ);
        X(destroy_symmetric_tridiagonal)(JP);
        X(destroy_symmetric_tridiagonal)(JQ);
        X(mpsm)('N', P, IDP, n, n);
        X(destroy_modified_plan)(P);
        alpha = 2;
        // u(x) = 1, v(x) = (1+x)*(2+x)
        u[0] = Y(sqrt)((FLT) 2);
        v[0] = Y(sqrt)((FLT) 1058);
        v[1] = -Y(sqrt)((FLT) 726);
        v[2] = Y(sqrt)((FLT) 48);
        P = X(plan_modified_laguerre_to_laguerre)(n, alpha, 1, u, 3, v, 0);
        X(mpsm)('N', P, IDP, n, n); // Should be the equivalent of the raising to x^2/(1+x)/(2+x)
        X(destroy_modified_plan)(P);
        alpha = 0;
        // u(x) = -x^2, v(x) = -(1+x)*(2+x)
        u[0] = -2;
        u[1] = 4;
        u[2] = -2;
        v[0] = -7;
        v[1] = 7;
        v[2] = -2;
        P = X(plan_modified_laguerre_to_laguerre)(n, alpha, 3, u, 3, v, 0);
        X(mpmm)('N', P, DP, n, n);
        X(destroy_modified_plan)(P);
        X(trmm)('N', n, IDP, n, DP, n, n);
        err = X(norm_2arg)(DP, Id, n*n)/X(norm_1arg)(Id, n*n);
        printf("Rational vs. raised recip. polynomial weight \t n = %3i |%20.2e ", n, (double) err);
        X(checktest)(err, Y(pow)(n+1, 4), checksum);
        free(Id);
        free(DP);
        free(IDP);
    }

    printf("\nTesting the accuracy of modified Hermite--Hermite transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 16; n < N; n *= 2) {
        // u(x) = v(x) = 1 + x^2 + x^4
        u[0] = v[0] = 2.995504568550877;
        u[1] = v[1] = 0;
        u[2] = v[2] = 3.7655850551068593;
        u[3] = v[3] = 0;
        u[4] = v[4] = 1.6305461589167827;
        P = X(plan_modified_hermite_to_hermite)(n, 5, u, 5, v, 0);
        FLT * DP = calloc(n*n, sizeof(FLT));
        FLT * IDP = calloc(n*n, sizeof(FLT));
        for (int j = 0; j < n; j++) {
            IDP[j+j*n] = DP[j+j*n] = 1;
            X(mpmv)('N', P, DP+j*n);
            X(mpsv)('N', P, IDP+j*n);
        }
        X(destroy_modified_plan)(P);
        err = X(norm_2arg)(DP, IDP, n*n)/X(norm_1arg)(DP, n*n);
        printf("Trivial weight r(x) = (1+x²+x⁴)/(1+x²+x⁴) \t n = %3i |%20.2e ", n, err);
        X(checktest)(err, Y(pow)(n+1, 3), checksum);
        free(DP);
        free(IDP);
    }
}
