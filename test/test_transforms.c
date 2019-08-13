#include "fasttransforms.h"
#include "ftutilities.h"

const int N = 2048;

int main(void) {
    int checksum = 0;
    double err;
    double * A, * B, * C;
    ft_tb_eigen_FMM * F;

    printf("\nTesting the accuracy of Chebyshev--Legendre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int normleg = 0; normleg <= 1; normleg++) {
            for (int normcheb = 0; normcheb <= 1; normcheb++) {
                A = plan_legendre_to_chebyshev(normleg, normcheb, n);
                B = plan_chebyshev_to_legendre(normcheb, normleg, n);
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
                err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                free(A);
                free(B);
            }
        }
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, err);
        ft_checktest(err, 2*sqrt(n), &checksum);
        free(C);
    }

    printf("\nTesting the accuracy of fast Chebyshev--Legendre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int normleg = 0; normleg <= 1; normleg++) {
            for (int normcheb = 0; normcheb <= 1; normcheb++) {
                F = ft_plan_legendre_to_chebyshev(normleg, normcheb, n);
                B = plan_chebyshev_to_legendre(normcheb, normleg, n);
                ft_bfmm('N', F, B, n, n);
                err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                ft_destroy_tb_eigen_FMM(F);
                free(B);
                F = ft_plan_chebyshev_to_legendre(normcheb, normleg, n);
                B = plan_legendre_to_chebyshev(normleg, normcheb, n);
                ft_bfmm('N', F, B, n, n);
                err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                ft_destroy_tb_eigen_FMM(F);
                free(B);
            }
        }
        err /= 2;
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, err);
        ft_checktest(err, 2*sqrt(n), &checksum);
        free(C);
    }

    double lambda, mu;

    printf("\nTesting the accuracy of ultraspherical--ultraspherical transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
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
                    A = plan_ultraspherical_to_ultraspherical(norm1, norm2, n, lambda, mu);
                    B = plan_ultraspherical_to_ultraspherical(norm2, norm1, n, mu, lambda);
                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    free(A);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, lambda, mu, err);
            ft_checktest(err, 4*pow(n, fabs(mu-lambda)), &checksum);
        }
        free(C);
    }

    printf("\nTesting the accuracy of fast ultraspherical--ultraspherical transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
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
                    F = ft_plan_ultraspherical_to_ultraspherical(norm1, norm2, n, lambda, mu);
                    B = plan_ultraspherical_to_ultraspherical(norm2, norm1, n, mu, lambda);
                    ft_bfmm('N', F, B, n, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    ft_destroy_tb_eigen_FMM(F);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, lambda, mu, err);
            ft_checktest(err, 4*pow(n, fabs(mu-lambda)), &checksum);
        }
        free(C);
    }

    double alpha, beta, gamma, delta;

    printf("\nTesting the accuracy of Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 8; cases++) {
            err = 0;
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
                    A = plan_jacobi_to_jacobi(norm1, norm2, n, alpha, beta, gamma, delta);
                    B = plan_jacobi_to_jacobi(norm2, norm1, n, gamma, delta, alpha, beta);
                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    free(A);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, alpha, beta, gamma, delta, err);
            ft_checktest(err, 32*pow(n, MAX(fabs(gamma-alpha), fabs(delta-beta))), &checksum);
        }
        A = plan_jacobi_to_jacobi(1, 1, n, 0.0, 0.0, -0.5, -0.5);
        B = plan_chebyshev_to_legendre(1, 1, n);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        free(A);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, 0.0, 0.0, -0.5, -0.5, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        A = plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, 0.0, 0.0);
        B = plan_legendre_to_chebyshev(1, 1, n);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        free(A);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, -0.5, -0.5, 0.0, 0.0, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        A = plan_jacobi_to_jacobi(1, 1, n, 0.0, 0.0, 0.5, 0.5);
        B = plan_ultraspherical_to_ultraspherical(1, 1, n, 1.0, 0.5);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        free(A);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, 0.0, 0.0, 0.5, 0.5, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        A = plan_jacobi_to_jacobi(1, 1, n, 0.5, 0.5, 0.0, 0.0);
        B = plan_ultraspherical_to_ultraspherical(1, 1, n, 0.5, 1.0);
        cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        free(A);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, 0.5, 0.5, 0.0, 0.0, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        free(C);
    }

    printf("\nTesting the accuracy of fast Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 8; cases++) {
            err = 0;
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
                    F = ft_plan_jacobi_to_jacobi(norm1, norm2, n, alpha, beta, gamma, delta);
                    B = plan_jacobi_to_jacobi(norm2, norm1, n, gamma, delta, alpha, beta);
                    ft_bfmm('N', F, B, n, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    ft_destroy_tb_eigen_FMM(F);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, alpha, beta, gamma, delta, err);
            ft_checktest(err, 32*pow(n, MAX(fabs(gamma-alpha), fabs(delta-beta))), &checksum);
        }
        F = ft_plan_jacobi_to_jacobi(1, 1, n, 0.0, 0.0, -0.5, -0.5);
        B = plan_chebyshev_to_legendre(1, 1, n);
        ft_bfmm('N', F, B, n, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        ft_destroy_tb_eigen_FMM(F);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, 0.0, 0.0, -0.5, -0.5, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        F = ft_plan_jacobi_to_jacobi(1, 1, n, -0.5, -0.5, 0.0, 0.0);
        B = plan_legendre_to_chebyshev(1, 1, n);
        ft_bfmm('N', F, B, n, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        ft_destroy_tb_eigen_FMM(F);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, -0.5, -0.5, 0.0, 0.0, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        F = ft_plan_jacobi_to_jacobi(1, 1, n, 0.0, 0.0, 0.5, 0.5);
        B = plan_ultraspherical_to_ultraspherical(1, 1, n, 1.0, 0.5);
        ft_bfmm('N', F, B, n, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        ft_destroy_tb_eigen_FMM(F);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, 0.0, 0.0, 0.5, 0.5, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        F = ft_plan_jacobi_to_jacobi(1, 1, n, 0.5, 0.5, 0.0, 0.0);
        B = plan_ultraspherical_to_ultraspherical(1, 1, n, 0.5, 1.0);
        ft_bfmm('N', F, B, n, n);
        err = ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
        ft_destroy_tb_eigen_FMM(F);
        free(B);
        printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, 0.5, 0.5, 0.0, 0.0, err);
        ft_checktest(err, 32*pow(n, 0.5), &checksum);
        free(C);
    }

    printf("\nTesting the accuracy of Laguerre--Laguerre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
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
                    beta = 2.5;
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = plan_laguerre_to_laguerre(norm1, norm2, n, alpha, beta);
                    B = plan_laguerre_to_laguerre(norm2, norm1, n, beta, alpha);
                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    free(A);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, alpha, beta, err);
            ft_checktest(err, 4*pow(n, fabs(alpha-beta)), &checksum);
        }
        free(C);
    }

    printf("\nTesting the accuracy of fast Laguerre--Laguerre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
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
                    beta = 2.5;
                    break;
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    F = ft_plan_laguerre_to_laguerre(norm1, norm2, n, alpha, beta);
                    B = plan_laguerre_to_laguerre(norm2, norm1, n, beta, alpha);
                    ft_bfmm('N', F, B, n, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    ft_destroy_tb_eigen_FMM(F);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, alpha, beta, err);
            ft_checktest(err, 4*pow(n, fabs(alpha-beta)), &checksum);
        }
        free(C);
    }

    printf("\nTesting the accuracy of Associated Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    A = plan_associated_jacobi_to_jacobi(0, 10, 1, 0.0, 0.0, 1.0, 1.0);
    printmat("Vc", "%17.16e", A, 10, 10);
    free(A);

    printf("\nTesting the accuracy of Konoplev--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    A = plan_konoplev_to_jacobi(10, 0.456, -0.25);
    printmat("Vc", "%17.16e", A, 10, 10);
    free(A);
    /*
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        C = calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
            A = plan_konoplev_to_jacobi(n, 0.0, -0.5);
            B = plan_konoplev_to_jacobi(n, 0.0, -0.5);
            B = plan_chebyshev_to_legendre(normcheb, normleg, n);
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
            err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
            free(A);
            free(B);
        printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, err);
        ft_checktest(err, 2*sqrt(n), &checksum);
        free(C);
    }
    */
    return checksum;
}
