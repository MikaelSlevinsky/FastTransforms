#include "fasttransforms.h"
#include "ftutilities.h"

const int N = 2048;

int main(void) {
    int checksum = 0;
    double err;
    double * A, * B, * C;

    printf("\nTesting the accuracy of Chebyshev--Legendre transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        C = (double *) calloc(n * n, sizeof(double));
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

    double lambda1, lambda2;

    printf("\nTesting the accuracy of ultraspherical--ultraspherical transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 4; cases++) {
            err = 0;
            switch (cases) {
                case 0:
                {
                    lambda1 = -0.125;
                    lambda2 = 0.125;
                    break;
                }
                case 1:
                {
                    lambda1 = 1.5;
                    lambda2 = 1.0;
                    break;
                }
                case 2:
                {
                    lambda1 = 0.25;
                    lambda2 = 1.25;
                    break;
                }
                case 3:
                {
                    lambda1 = 0.5;
                    lambda2 = 2.5;
                    break;
                }
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = plan_ultraspherical_to_ultraspherical(norm1, norm2, n, lambda1, lambda2);
                    B = plan_ultraspherical_to_ultraspherical(norm2, norm1, n, lambda2, lambda1);
                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    free(A);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f) → (%+1.2f): \t\t |%20.2e ", n, n, lambda1, lambda2, err);
            ft_checktest(err, 4*pow(n, fabs(lambda2-lambda1)), &checksum);
        }
        free(C);
    }

    double alpha, beta, gamma, delta;

    printf("\nTesting the accuracy of Jacobi--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 64; n < N; n *= 2) {
        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
        for (int cases = 0; cases < 8; cases++) {
            err = 0;
            switch (cases) {
                case 0:
                {
                    alpha = 0.0;
                    beta = -0.5;
                    gamma = -0.5;
                    delta = -0.5;
                    break;
                }
                case 1:
                {
                    alpha = 0.1;
                    beta = 0.2;
                    gamma = 0.3;
                    delta = 0.4;
                    break;
                }
                case 2:
                {
                    alpha = 1.0;
                    beta = 0.5;
                    gamma = 0.5;
                    delta = 0.25;
                    break;
                }
                case 3:
                {
                    alpha = -0.25;
                    beta = -0.75;
                    gamma = 0.25;
                    delta = 0.75;
                    break;
                }
                case 4:
                {
                    alpha = 0.0;
                    beta = 1.0;
                    gamma = -0.5;
                    delta = 0.5;
                    break;
                }
                case 5:
                {
                    alpha = 0.0;
                    beta = -0.5;
                    gamma = -0.5;
                    delta = -0.25;
                    break;
                }
                case 6:
                {
                    alpha = -0.5;
                    beta = 0.5;
                    gamma = -0.5;
                    delta = 0.0;
                    break;
                }
                case 7:
                {
                    alpha = 0.5;
                    beta = -0.5;
                    gamma = -0.5;
                    delta = 0.0;
                    break;
                }
            }
            for (int norm1 = 0; norm1 <= 1; norm1++) {
                for (int norm2 = 0; norm2 <= 1; norm2++) {
                    A = eigenplan_jacobi_to_jacobi(norm1, norm2, n, alpha, beta, gamma, delta);
                    B = eigenplan_jacobi_to_jacobi(norm2, norm1, n, gamma, delta, alpha, beta);
                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);
                    err += ft_norm_2arg(B, C, n*n)/ft_norm_1arg(C, n*n);
                    free(A);
                    free(B);
                }
            }
            printf("(n×n) = (%4ix%4i), (%+1.2f, %+1.2f) → (%+1.2f, %+1.2f): \t |%20.2e ", n, n, alpha, beta, gamma, delta, err);
            ft_checktest(err, 32*pow(n, MAX(fabs(gamma-alpha), fabs(delta-beta))), &checksum);
        }
        free(C);
    }

    printf("\nTesting the accuracy of Konoplev--Jacobi transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    A = eigenplan_konoplev_to_jacobi(16, 0.456, -0.25);
    printmat("Vc", "%17.16e", A, 16, 16);
    free(A);
    /*
    for (int n = 64; n < N; n *= 2) {
        err = 0;
        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;
            A = eigenplan_konoplev_to_jacobi(n, 0.0, -0.5);
            B = eigenplan_konoplev_to_jacobi(n, 0.0, -0.5);
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
