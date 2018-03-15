#include <cblas.h>
#include "../src/transforms.h"
#include "utilities.h"

const int N = 4097;

int main(void) {
    static double * A, * B, * C;

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

                printf("\t\t Normalization of SRC:TRG = %d:%d. The relative Frobenius norm error is: %17.16e.\n", normleg, normcheb, vecnorm_2arg(B, C, n, n)/vecnorm_1arg(C, n, n));

                free(A);
                free(B);
            }
        }
        free(C);
    }

    double lambda1 = 1.5, lambda2 = 1.0;

    printf("\n\nTesting the accuracy of ultraspherical--ultraspherical transforms.\n");
    for (int n = 64; n < N; n *= 2) {

        printf("\tTransform dimensions: %d x %d.\n", n, n);

        C = (double *) calloc(n * n, sizeof(double));
        for (int i = 0; i < n; i++)
            C[i+n*i] = 1.0;

        for (int normultra1 = 0; normultra1 <= 1; normultra1++) {
            for (int normultra2 = 0; normultra2 <= 1; normultra2++) {

                A = plan_ultra2ultra(normultra1, normultra2, n, lambda1, lambda2);
                B = plan_ultra2ultra(normultra2, normultra1, n, lambda2, lambda1);

                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);

                printf("\t\t Normalization of SRC:TRG = %d:%d. The relative Frobenius norm error is: %17.16e.\n", normultra1, normultra2, vecnorm_2arg(B, C, n, n)/vecnorm_1arg(C, n, n));

                free(A);
                free(B);
            }
        }
        free(C);
    }

    return 0;
}
