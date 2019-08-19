void Y(test_transforms)(int * checksum, int N) {
    FLT err;
    FLT * Id, * B;
    X(tb_eigen_FMM) * A;

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

    printf("\nTesting the accuracy of interrelated transforms.\n\n");
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
}
