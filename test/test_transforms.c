#include "utilities.h"

const int N = 2049;

int main(void) {
    double * A, * B, * C;

    printf("\n\nTesting the accuracy of Chebyshev--Legendre transforms.\n");
    for (int n = 64; n < N; n *= 2) {

        printf("\tTransform dimensions: %d x %d.\n", n, n);

        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;

        for (int normleg = 0; normleg <= 1; normleg++) {
            for (int normcheb = 0; normcheb <= 1; normcheb++) {

                A = plan_leg2cheb(normleg, normcheb, n);
                B = plan_cheb2leg(normcheb, normleg, n);

                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);

                printf("\t\tNormalization of SRC:TRG = %d:%d. The relative Frobenius norm error is: %1.2e.\n", normleg, normcheb, vecnorm_2arg(B, C, n, n)/vecnorm_1arg(C, n, n));

                free(A);
                free(B);
            }
        }
        free(C);
    }

    double lambda1, lambda2;

    printf("\n\nTesting the accuracy of ultraspherical--ultraspherical transforms.\n");
    for (int n = 64; n < N; n *= 2) {

        printf("\tTransform dimensions: %d x %d.\n", n, n);

        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;

        for (int cases = 0; cases < 3; cases++) {
            switch (cases) {
                case 0:
                {
                    lambda1 = 1.5;
                    lambda2 = 1.0;
                    break;
                }
                case 1:
                {
                    lambda1 = 0.25;
                    lambda2 = 1.25;
                    break;
                }
                case 2:
                {
                    lambda1 = 0.5;
                    lambda2 = 2.5;
                    break;
                }
            }
            printf("\t\tUltraspherical parameters (%1.2f) → (%1.2f): \n", lambda1, lambda2);

            for (int normultra1 = 0; normultra1 <= 1; normultra1++) {
                for (int normultra2 = 0; normultra2 <= 1; normultra2++) {

                    A = plan_ultra2ultra(normultra1, normultra2, n, lambda1, lambda2);
                    B = plan_ultra2ultra(normultra2, normultra1, n, lambda2, lambda1);

                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);

                    printf("\t\t\tNormalization of SRC:TRG = %d:%d. The relative Frobenius norm error is: %1.2e.\n", normultra1, normultra2, vecnorm_2arg(B, C, n, n)/vecnorm_1arg(C, n, n));

                    free(A);
                    free(B);
                }
            }
        }
        free(C);
    }

    double alpha, beta, gamma;

    printf("\n\nTesting the accuracy of Jacobi--Jacobi transforms.\n");
    for (int n = 64; n < N; n *= 2) {

        printf("\tTransform dimensions: %d x %d.\n", n, n);

        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;

        for (int cases = 0; cases < 8; cases++) {
            switch (cases) {
                case 0:
                {
                    alpha = 0.0;
                    beta = -0.5;
                    gamma = -0.5;
                    break;
                }
                case 1:
                {
                    alpha = 0.1;
                    beta = 0.2;
                    gamma = 0.3;
                    break;
                }
                case 2:
                {
                    alpha = 1.0;
                    beta = 0.5;
                    gamma = 0.5;
                    break;
                }
                case 3:
                {
                    alpha = -0.25;
                    beta = -0.75;
                    gamma = 0.25;
                    break;
                }
                case 4:
                {
                    alpha = 1.0;
                    beta = 0.0;
                    gamma = -0.5;
                    break;
                }
                case 5:
                {
                    alpha = 0.0;
                    beta = -0.5;
                    gamma = -0.5;
                    break;
                }
                case 6:
                {
                    alpha = -0.5;
                    beta = 0.5;
                    gamma = -0.5;
                    break;
                }
                case 7:
                {
                    alpha = 0.5;
                    beta = -0.5;
                    gamma = -0.5;
                    break;
                }
            }
            printf("\t\tJacobi parameters (%1.2f, %1.2f) → (%1.2f, %1.2f): \n", alpha, beta, gamma, beta);

            for (int normjac1 = 0; normjac1 <= 1; normjac1++) {
                for (int normjac2 = 0; normjac2 <= 1; normjac2++) {

                    A = plan_jac2jac(normjac1, normjac2, n, alpha, beta, gamma);
                    B = plan_jac2jac(normjac2, normjac1, n, gamma, beta, alpha);

                    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);

                    printf("\t\t\tNormalization of SRC:TRG = %d:%d. The relative Frobenius norm error is: %1.2e.\n", normjac1, normjac2, vecnorm_2arg(B, C, n, n)/vecnorm_1arg(C, n, n));

                    free(A);
                    free(B);
                }
            }
        }
        free(C);
    }

    return 0;
}
