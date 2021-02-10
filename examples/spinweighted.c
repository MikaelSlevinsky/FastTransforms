#include <complex.h>
#include <fasttransforms.h>
#include <ftutilities.h>

typedef struct {
    double x;
    double y;
    double z;
} double3;

double dot3(double3 x, double3 y) {return x.x*y.x+x.y*y.y+x.z*y.z;};

double3 r(double theta, double phi) {return (double3) {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};};

/*!
  \example spinweighted.c
  This example plays with analysis of:
  \f[
  f(r) = e^{{\rm i} k\cdot r},
  \f]
  for some \f$k\in\mathbb{R}^3\f$ and where \f$r\in\mathbb{S}^2\f$, using spin-0 spherical harmonics. It applies ð, the spin-raising operator, both on the spin-0 coefficients as well as the original function, followed by a spin-1 analysis to compare coefficients.
*/
int main(void) {
    printf("This example plays with analysis of\n");
    printf("\n");
    printf("\t"MAGENTA("f(r) = exp(i k⋅r)")",\n");
    printf("\n");
    printf("for some "MAGENTA("k ∈ ℝ³")" and where "MAGENTA("r ∈ 𝕊²")", ");
    printf("using spin-0 spherical harmonics.\n");
    printf("\n");
    printf("It applies "MAGENTA("ð")", the spin-raising operator, both on the spin-0 coefficients\n");
    printf("as well as the original function, followed by a spin-1 analysis to compare coefficients.\n\n");
    printf("This is accomplished by using \n");
    printf("\t"CYAN("ft_plan_spinsph_analysis")" and "CYAN("ft_execute_spinsph_analysis")",\n");
    printf("and \t"CYAN("ft_plan_spinsph2fourier")" and "CYAN("ft_execute_fourier2spinsph")".\n");

    char * FMT = "%1.3f";

    int N = 5;
    int M = 2*N-1;

    printf("\n\n"MAGENTA("N = %i")", and "MAGENTA("M = %i")"\n\n", N, M);

    double theta[N], phi[M];
    ft_complex F[N*M], G[N*M];
    double complex * FC = (double complex *) F;
    double complex * GC = (double complex *) G;

    for (int n = 0; n < N; n++)
        theta[n] = (n+0.5)*M_PI/N;
    for (int m = 0; m < M; m++)
        phi[m] = 2.0*M_PI*m/M;

    printmat("Colatitudinal grid "MAGENTA("θ"), FMT, theta, N, 1);
    printf("\n");
    printmat("Longitudinal grid "MAGENTA("φ"), FMT, phi, 1, M);
    printf("\n");

    double3 k = {2.0/7.0, 3.0/7.0, 6.0/7.0};

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            FC[n+N*m] = cexp(I*dot3(k, r(theta[n], phi[m])));

    printf("On the tensor product grid, the function "MAGENTA("exp(i k⋅r)")" is:\n\n");

    printmat("F", FMT, (double *) F, 2*N, M);
    printf("\n");

    ft_spin_harmonic_plan * P = ft_plan_spinsph2fourier(N, 0);
    ft_spinsphere_fftw_plan * PA = ft_plan_spinsph_analysis(N, M, 0);

    ft_execute_spinsph_analysis('N', PA, F, N, M);
    ft_execute_fourier2spinsph('N', P, F, N, M);

    printf("Its spin-0 spherical harmonic coefficients are:\n\n");

    printmat("U⁰", FMT, (double *) F, 2*N, M);
    printf("\n");

    double nrm = ft_norm_1arg((double *) F, 2*N*M);

    printf("The 2-norm of its coefficients is: \t\t %1.8f.\n", nrm);
    printf("This compares favourably to the exact result: \t %1.8f.\n\n", sqrt(4*M_PI));

    for (int n = 1; n < N; n++)
        GC[n-1] = sqrt(n*(n+1))*FC[n];
    for (int m = 1; m <= M/2; m++)
        for (int n = 0; n < N; n++) {
            GC[n+N*(2*m-1)] = -sqrt((n+m)*(n+m+1))*FC[n+N*(2*m-1)];
            GC[n+N*(2*m)] = sqrt((n+m)*(n+m+1))*FC[n+N*(2*m)];
        }

    printf("Spin can be incremented by applying ð, either on the spin-0 coefficients:\n\n");

    printmat("U¹coefficients", FMT, (double *) G, 2*N, M);
    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            FC[n+N*m] = -(k.x*(I*cos(theta[n])*cos(phi[m]) + sin(phi[m])) + k.y*(I*cos(theta[n])*sin(phi[m])-cos(phi[m])) - I*k.z*sin(theta[n]))*cexp(I*dot3(k, r(theta[n], phi[m])));

    printf("or on the original function through analysis with spin-1 spherical harmonics:\n\n");

    ft_destroy_spin_harmonic_plan(P);
    ft_destroy_spinsphere_fftw_plan(PA);

    P = ft_plan_spinsph2fourier(N, 1);
    PA = ft_plan_spinsph_analysis(N, M, 1);

    ft_execute_spinsph_analysis('N', PA, F, N, M);
    ft_execute_fourier2spinsph('N', P, F, N, M);

    printmat("U¹sampling", FMT, (double *) F, 2*N, M);
    printf("\n");

    nrm = ft_norm_1arg((double *) F, 2*N*M);

    printf("The 2-norm of the spin-1 coefficients is: \t %1.8f.\n", nrm);
    printf("This is also quite close to the exact result: \t %1.8f.\n\n", sqrt(8.0*M_PI/3.0*dot3(k, k)));

    ft_destroy_spin_harmonic_plan(P);
    ft_destroy_spinsphere_fftw_plan(PA);

    return 0;
}
