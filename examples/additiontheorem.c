#include <stdio.h>
#include "fasttransforms.h"

void printmat(char * MAT, double * A, int n, int m);
double dot3(double * x, double * y);
void normalize3(double * x);
double P4(double x);
double * z(double theta, double phi);

int main(void) {
    printf("\nThis example confirms numerically that\n");
    printf("\n");
    printf("\t[P₄(z⋅y) - P₄(x⋅y)]/(z⋅y - x⋅y),\n");
    printf("\n");
    printf("is actually a degree-3 polynomial on 𝕊², where P₄ is the degree-4\n");
    printf("Legendre polynomial, and x,y,z ∈ 𝕊².\n");
    printf("To verify, we sample the function on a 5×9 tensor product grid\n");
    printf("at equispaced points-in-angle defined by:\n");
    printf("\n");
    printf("\tθₙ = (n+1/2)π/N, for 0 ≤ n < N,\n");
    printf("\n");
    printf("and\n");
    printf("\n");
    printf("\tφₘ = 2π m/M, for 0 ≤ m < M;\n");
    printf("\n");
    printf("we convert the function samples to Fourier coefficients using\n");
    printf("`plan_sph_analysis` and `execute_sph_analysis`; and finally, we transform\n");
    printf("the Fourier coefficients to spherical harmonic coefficients using\n");
    printf("`plan_sph2fourier` and `execute_fourier2sph`.\n");
    printf("\n");
    printf("In the basis of spherical harmonics, it is plain to see the\n");
    printf("addition theorem in action, since P₄(x⋅y) should only consist of\n");
    printf("exact-degree-4 harmonics.\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    SphericalHarmonicPlan * P;
    SphereFFTWPlan * PA;

    double x[] = {0.0,0.0,1.0};
    double y[] = {0.123,0.456,0.789};
    normalize3(y);

    int N = 5;
    int M = 2*N-1;

    printf("\n\nN = %i, and M = %i\n\n", N, M);

    PA = plan_sph_analysis(N, M);
    P = plan_sph2fourier(N);

    double * theta = malloc(N*sizeof(double));
    double * phi = malloc(M*sizeof(double));

    for (int n = 0; n < N; n++)
        theta[n] = (n+0.5)*M_PI/N;
    for (int m = 0; m < M; m++)
        phi[m] = 2.0*M_PI*m/M;

    for (int n = 0; n < N; n++)
        printf("Colatitudinal grid theta[%i] = %1.3e\n", n, theta[n]);
    printf("\n");
    for (int m = 0; m < M; m++)
        printf("Longitudinal grid phi[%i] = %1.3e\n", m, phi[m]);
    printf("\n");

    printf("Arbitrarily, we place x at the North pole: x = (%1.2e,%1.2e,%1.2e)ᵀ.\n\n",x[0],x[1],x[2]);
    printf("Another vector is completely free: y = (%1.2e,%1.2e,%1.2e)ᵀ.\n\n",y[0],y[1],y[2]);
    printf("Thus z ∈ 𝕊² is our variable vector.\n\n");

    double * F = calloc(N*M, sizeof(double));

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = (P4(dot3(z(theta[n], phi[m]), y)) - P4(dot3(x, y)))/(dot3(z(theta[n], phi[m]), y) - dot3(x, y));

    printf("On the tensor product grid, our function samples are:\n\n");

    printmat("F", F, N, M);
    printf("\n");

    execute_sph_analysis(PA, F, N, M);
    execute_fourier2sph(P, F, N, M);

    printf("Its spherical harmonic coefficients demonstrate that it is degree-3:\n\n");

    printmat("U3", F, N, M);

    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = P4(dot3(z(theta[n], phi[m]), y));

    printf("Similarly, on the tensor product grid, the Legendre polynomial P₄(z⋅y) is:\n\n");

    printmat("F", F, N, M);
    printf("\n");

    execute_sph_analysis(PA, F, N, M);
    execute_fourier2sph(P, F, N, M);

    printf("Its spherical harmonic coefficients demonstrate that it is exact-degree-4:\n\n");

    printmat("U4", F, N, M);

    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = P4(dot3(z(theta[n], phi[m]), x));

    printf("Finally, the Legendre polynomial P₄(z⋅x) is aligned with the grid:\n\n");

    printmat("F", F, N, M);
    printf("\n");

    execute_sph_analysis(PA, F, N, M);
    execute_fourier2sph(P, F, N, M);

    printf("It only has one nonnegligible spherical harmonic coefficient.\n");
    printf("Can you spot it?\n\n");

    printmat("U4", F, N, M);

    printf("\n");

    printf("That nonnegligible coefficient should be approximately √(2π/(4+1/2)),\n");
    printf("since the convention in this library is to orthonormalize.\n");

    freeSphericalHarmonicPlan(P);
    freeSphereFFTWPlan(PA);
    free(theta);
    free(phi);
    free(F);

    return 0;
}

void printmat(char * MAT, double * A, int n, int m) {
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            printf("%s[%d][%d] = %17.16f\n", MAT, i, j, A[i+n*j]);
}

double dot3(double * x, double * y) {return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];};

void normalize3(double * x) {
    double nrm = sqrt(dot3(x, x));
    x[0] /= nrm; x[1] /= nrm; x[2] /= nrm;
}

double P4(double x) {return (35.0*pow(x, 4)-30.0*pow(x, 2)+3.0)/8.0;};

double * z(double theta, double phi) {
    double * ret = malloc(3*sizeof(double));
    ret[0] = sin(theta)*cos(phi);
    ret[1] = sin(theta)*sin(phi);
    ret[2] = cos(theta);
    return ret;
}
