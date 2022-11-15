#include <fasttransforms.h>
#include <ftutilities.h>

double f(double x, double y) {return (pow(x, 3))/(x*x+y*y-0.25);};

/*!
  \example annulus.c
  In this example, we explore integration of the function:
  \f[
  f(x,y) = \frac{x^3}{x^2+y^2-\frac{1}{4}},
  \f]
  over the annulus defined by \f$\{(r,\theta) : \frac{2}{3} < r < 1, 0 < \theta < 2\pi\}\f$. We will calculate the integral:
  \f[
  \int_0^{2\pi}\int_{\frac{2}{3}}^1 f(r\cos\theta,r\sin\theta)^2r{\rm\,d}r{\rm\,d}\theta,
  \f]
  by analyzing the function in an annulus polynomial series.
*/
int main(void) {
    printf("In this example, we explore square integration of a function over \n");
    printf("the annulus with parameter "MAGENTA("ρ = 2/3")". We analyze the function:\n");
    printf("\n");
    printf("\t"MAGENTA("f(x,y) = x³/(x²-y²-1/4)")",\n");
    printf("\n");
    printf("on an "MAGENTA("N×M")" tensor product grid defined by:\n");
    printf("\n");
    printf("\t"MAGENTA("rₙ = √{cos²[(n+1/2)π/4N] + ρ²sin²[(n+1/2)π/4N]}")", for "MAGENTA("0 ≤ n < N")",\n");
    printf("\n");
    printf("and\n");
    printf("\n");
    printf("\t"MAGENTA("θₘ = 2π m/M")", for "MAGENTA("0 ≤ m < M")";\n");
    printf("\n");
    printf("we convert the function samples to Chebyshev×Fourier coefficients using\n");
    printf(CYAN("ft_plan_annulus_analysis")" and "CYAN("ft_execute_annulus_analysis")"; and finally, we transform\n");
    printf("the Chebyshev×Fourier coefficients to annulus harmonic coefficients using\n");
    printf(CYAN("ft_plan_ann2cxf")" and "CYAN("ft_execute_cxf2ann")".\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    char * FMT = "%1.3f";

    int N = 8;
    int M = 4*N-3;

    printf("\n\n"MAGENTA("N = %i")", and "MAGENTA("M = %i")"\n\n", N, M);

    double rho = 2.0/3.0;
    double r[N], theta[M], F[4*N*N];

    for (int n = 0; n < N; n++) {
        double t = (N-n-0.5)*M_PI/(2*N);
        double ct = sin(t);
        double st = cos(t);
        r[n] = sqrt(ct*ct+rho*rho*st*st);
    }
    for (int m = 0; m < M; m++)
        theta[m] = 2.0*M_PI*m/M;

    printmat("Radial grid "MAGENTA("r"), FMT, r, N, 1);
    printf("\n");
    printmat("Azimuthal grid "MAGENTA("θ"), FMT, theta, 1, M);
    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = f(r[n]*cos(theta[m]), r[n]*sin(theta[m]));

    printf("On the tensor product grid, our function samples are:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    double alpha = 0.0, beta = 0.0, gamma = 0.0;
    ft_harmonic_plan * P = ft_plan_ann2cxf(N, alpha, beta, gamma, rho);
    ft_annulus_fftw_plan * PA = ft_plan_annulus_analysis(N, M, rho, FT_FFTW_FLAGS);

    ft_execute_annulus_analysis('N', PA, F, N, M);
    ft_execute_cxf2ann('N', P, F, N, M);

    printf("Its annulus polynomial coefficients are:\n\n");

    printmat("U", FMT, F, N, M);
    printf("\n");

    printf("The annulus polynomial coefficients are useful for integration.\n");
    printf("The integral of "MAGENTA("[f(x,y)]^2")" over the annulus is\n");
    printf("approximately the square of the 2-norm of the coefficients, \n\t");
    double val = pow(ft_norm_1arg(F, N*M), 2);
    printf("%1.16f", val);
    printf(".\n");
    printf("This compares favourably to the exact result, \n\t");
    double tval = 5.0*M_PI/8.0*(1675.0/4536.0+9.0*log(3.0)/32.0-3.0*log(7.0)/32.0);
    printf("%1.16f", tval);
    printf(".\n");
    printf("The relative error in the integral is %4.2e.\n", fabs(val-tval)/fabs(tval));
    printf("This error can be improved upon by increasing "MAGENTA("N")" and "MAGENTA("M")".\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_annulus_fftw_plan(PA);

    return 0;
}
