#include <fasttransforms.h>
#include <ftutilities.h>

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

double3 z(double theta, double phi) {return (double3) {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};};

/*!
  \example additiontheorem.c
  This example confirms numerically that:
  \f[
  f(z) = \frac{P_n(z\cdot y) - P_n(x\cdot y)}{z\cdot y - x\cdot y},
  \f]
  is actually a degree-\f$n-1\f$ polynomial on \f$\mathbb{S}^2\f$, where \f$P_n\f$ is the degree-\f$n\f$ Legendre polynomial, and \f$x,y,z \in \mathbb{S}^2\f$.
*/
int main(void) {
    printf("This example confirms numerically that\n");
    printf("\n");
    printf("\t"MAGENTA("[Pₙ(z⋅y) - Pₙ(x⋅y)]/(z⋅y - x⋅y)")",\n");
    printf("\n");
    printf("is actually a degree-(N-1) polynomial on "MAGENTA("𝕊²")", where "MAGENTA("Pₙ")" is the degree-N\n");
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
    printf("addition theorem in action, since "MAGENTA("Pₙ(x⋅y)")" should only consist of\n");
    printf("exact-degree-N harmonics.\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    char * FMT = "%1.3f";

    int N = 5;
    int M = 2*N-1;

    printf("\n\n"MAGENTA("N = %i")", and "MAGENTA("M = %i")"\n\n", N, M);

    double theta[N], phi[M], F[N*M];

    for (int n = 0; n < N; n++)
        theta[n] = (n+0.5)*M_PI/N;
    for (int m = 0; m < M; m++)
        phi[m] = 2.0*M_PI*m/M;

    printmat("Colatitudinal grid "MAGENTA("θ"), FMT, theta, N, 1);
    printf("\n");
    printmat("Longitudinal grid "MAGENTA("φ"), FMT, phi, 1, M);
    printf("\n");

    double3 x = {0.0,0.0,1.0};
    double3 y = {0.123,0.456,0.789};
    normalize3(&y);

    printf("Arbitrarily, we place "MAGENTA("x")" at the North pole: "MAGENTA("x = (%1.3f,%1.3f,%1.3f)ᵀ")".\n\n", x.x, x.y, x.z);
    printf("Another vector is completely free: "MAGENTA("y = (%1.3f,%1.3f,%1.3f)ᵀ")".\n\n", y.x, y.y, y.z);
    printf("Thus "MAGENTA("z ∈ 𝕊²")" is our variable vector.\n\n");

    double A[N], B[N], C[N+1], c[N], ones[N*M], pts[N*M];
    for (int k = 0; k < N; k++) {
        A[k] = (2*k+1.0)/(k+1.0);
        B[k] = 0.0;
        C[k] = k/(k+1.0);
        c[k] = 0.0;
    }
    C[N] = N/(N+1.0);
    c[N-1] = 1.0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            pts[n+N*m] = dot3(z(theta[n], phi[m]), y);
            ones[n+N*m] = 1.0;
        }
    ft_orthogonal_polynomial_clenshaw(N, c, 1, A, B, C, N*M, pts, ones, F);

    printf("On the tensor product grid, the Legendre polynomial "MAGENTA("Pₙ(z⋅y)")" is:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_harmonic_plan * P = ft_plan_sph2fourier(N);
    ft_sphere_fftw_plan * PA = ft_plan_sph_analysis(N, M);

    ft_execute_sph_analysis(PA, F, N, M);
    ft_execute_fourier2sph('N', P, F, N, M);

    printf("Its spherical harmonic coefficients demonstrate that it is exact-degree-%i:\n\n", N-1);

    printmat("U_{N-1}", FMT, F, N, M);
    printf("\n");

    double xy = dot3(x, y), Pnxy;
    ft_orthogonal_polynomial_clenshaw(N, c, 1, A, B, C, 1, &xy, ones, &Pnxy);
    ft_orthogonal_polynomial_clenshaw(N, c, 1, A, B, C, N*M, pts, ones, F);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = (F[n+N*m] - Pnxy)/(dot3(z(theta[n], phi[m]), y) - dot3(x, y));

    printf("Similarly, on the tensor product grid, our function samples are:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_sph_analysis(PA, F, N, M);
    ft_execute_fourier2sph('N', P, F, N, M);

    printf("Its spherical harmonic coefficients demonstrate that it is degree-(%i-1):\n\n", N-1);

    printmat("U_{N-2}", FMT, F, N, M);
    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            pts[n+N*m] = dot3(z(theta[n], phi[m]), x);
    ft_orthogonal_polynomial_clenshaw(N, c, 1, A, B, C, N*M, pts, ones, F);

    printf("Finally, the Legendre polynomial "MAGENTA("Pₙ(z⋅x)")" is aligned with the grid:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_sph_analysis(PA, F, N, M);
    ft_execute_fourier2sph('N', P, F, N, M);

    printf("It only has one nonnegligible spherical harmonic coefficient.\n");
    printf("Can you spot it?\n\n");

    printmat("U_{N-1}", FMT, F, N, M);
    printf("\n");

    printf("That nonnegligible coefficient should be approximately "MAGENTA("√(2π/(%i+1/2))")",\n", N-1);
    printf("since the convention in this library is to orthonormalize.\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_sphere_fftw_plan(PA);

    return 0;
}
