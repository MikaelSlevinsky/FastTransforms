#include "fasttransforms.h"
#include "ftutilities.h"

typedef struct {
    double x;
    double y;
    double z;
} double3;

double dot3(double3 x, double3 y) {return x.x*y.x+x.y*y.y+x.z*y.z;};

void normalize3(double3 * x) {
    double nrm = sqrt(dot3(* x, * x));
    x->x /= nrm; x->y /= nrm; x->z /= nrm;
}

double P4(double x) {double x2 = x*x; return ((35.0*x2-30.0)*x2+3.0)/8.0;};

double3 z(double theta, double phi) {return (double3) {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};};

int main(void) {
    printf("\nThis example confirms numerically that\n");
    printf("\n");
    printf("\t"MAGENTA("[P₄(z⋅y) - P₄(x⋅y)]/(z⋅y - x⋅y)")",\n");
    printf("\n");
    printf("is actually a degree-3 polynomial on "MAGENTA("𝕊²")", where "MAGENTA("P₄")" is the degree-4\n");
    printf("Legendre polynomial, and "MAGENTA("x,y,z ∈ 𝕊²")".\n");
    printf("To verify, we sample the function on an "MAGENTA("N×M")" tensor product grid\n");
    printf("at equispaced points-in-angle defined by:\n");
    printf("\n");
    printf("\t"MAGENTA("θₙ = (n+1/2)π/N")", for "MAGENTA("0 ≤ n < N")",\n");
    printf("\n");
    printf("and\n");
    printf("\n");
    printf("\t"MAGENTA("φₘ = 2π m/M")", for "MAGENTA("0 ≤ m < M")";\n");
    printf("\n");
    printf("we convert the function samples to Fourier coefficients using\n");
    printf(CYAN("ft_plan_sph_analysis")" and "CYAN("ft_execute_sph_analysis")"; and finally, we transform\n");
    printf("the Fourier coefficients to spherical harmonic coefficients using\n");
    printf(CYAN("ft_plan_sph2fourier")" and "CYAN("ft_execute_fourier2sph")".\n");
    printf("\n");
    printf("In the basis of spherical harmonics, it is plain to see the\n");
    printf("addition theorem in action, since "MAGENTA("P₄(x⋅y)")" should only consist of\n");
    printf("exact-degree-4 harmonics.\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    char * FMT = "%1.3f";

    ft_harmonic_plan * P;
    ft_sphere_fftw_plan * PA;

    double3 x = {0.0,0.0,1.0};
    double3 y = {0.123,0.456,0.789};
    normalize3(&y);

    int N = 5;
    int M = 2*N-1;

    printf("\n\n"MAGENTA("N = %i")", and "MAGENTA("M = %i")"\n\n", N, M);

    PA = ft_plan_sph_analysis(N, M);
    P = ft_plan_sph2fourier(N);

    double theta[N], phi[M], F[N*M];

    for (int n = 0; n < N; n++)
        theta[n] = (n+0.5)*M_PI/N;
    for (int m = 0; m < M; m++)
        phi[m] = 2.0*M_PI*m/M;

    printmat("Colatitudinal grid "MAGENTA("θ"), FMT, theta, N, 1);
    printf("\n");
    printmat("Longitudinal grid "MAGENTA("φ"), FMT, phi, 1, M);
    printf("\n");

    printf("Arbitrarily, we place "MAGENTA("x")" at the North pole: "MAGENTA("x = (%1.3f,%1.3f,%1.3f)ᵀ")".\n\n", x.x, x.y, x.z);
    printf("Another vector is completely free: "MAGENTA("y = (%1.3f,%1.3f,%1.3f)ᵀ")".\n\n", y.x, y.y, y.z);
    printf("Thus "MAGENTA("z ∈ 𝕊²")" is our variable vector.\n\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = (P4(dot3(z(theta[n], phi[m]), y)) - P4(dot3(x, y)))/(dot3(z(theta[n], phi[m]), y) - dot3(x, y));

    printf("On the tensor product grid, our function samples are:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_sph_analysis(PA, F, N, M);
    ft_execute_fourier2sph(P, F, N, M);

    printf("Its spherical harmonic coefficients demonstrate that it is degree-3:\n\n");

    printmat("U3", FMT, F, N, M);

    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = P4(dot3(z(theta[n], phi[m]), y));

    printf("Similarly, on the tensor product grid, the Legendre polynomial "MAGENTA("P₄(z⋅y)")" is:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_sph_analysis(PA, F, N, M);
    ft_execute_fourier2sph(P, F, N, M);

    printf("Its spherical harmonic coefficients demonstrate that it is exact-degree-4:\n\n");

    printmat("U4", FMT, F, N, M);

    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = P4(dot3(z(theta[n], phi[m]), x));

    printf("Finally, the Legendre polynomial "MAGENTA("P₄(z⋅x)")" is aligned with the grid:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_sph_analysis(PA, F, N, M);
    ft_execute_fourier2sph(P, F, N, M);

    printf("It only has one nonnegligible spherical harmonic coefficient.\n");
    printf("Can you spot it?\n\n");

    printmat("U4", FMT, F, N, M);

    printf("\n");

    printf("That nonnegligible coefficient should be approximately "MAGENTA("√(2π/(4+1/2))")",\n");
    printf("since the convention in this library is to orthonormalize.\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_sphere_fftw_plan(PA);

    return 0;
}
