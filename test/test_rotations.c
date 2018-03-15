#include "../src/rotations.h"
#include "utilities.h"

const int N = 2049;

int main(void) {
    static double * A, * B;
    RotationPlan * RP;
    double nrm;

    printf("\n\nTesting the computation of the spherical harmonic Givens rotations.\n");
    for (int n = 64; n < N; n *= 2) {
        printf("\tDegree: %d.\n", n);

        RP = plan_rotsphere(n);

        printf("\t\tThe 2-norm relative error in sqrt(s^2+c^2) is: %17.16e.\n", rotnorm(RP)/sqrt(n*(n+1)/2));

        A = (double *) calloc(n, sizeof(double));
        B = (double *) calloc(n, sizeof(double));

        nrm = 0.0;
        for (int m = 2; m < n; m++) {
            for (int i = 0; i < n-m; i++) {
                A[i] = 1.0;
                B[i] = 1.0;
            }
            for (int i = n-m; i < n; i++) {
                A[i] = 0.0;
                B[i] = 0.0;
            }
            kernel1_sph_hi2lo(RP, A, m);
            kernel1_sph_lo2hi(RP, A, m);
            nrm += pow(vecnorm_2arg(A, B, n, 1)/vecnorm_1arg(B, n, 1), 2);
        }
        printf("\t\tThe 2-norm relative error in the rotations: %17.16e.\n", sqrt(nrm));
        free(A);
        free(B);
        free(RP);
    }

    return 0;
}
