#include <fasttransforms.h>
#include <ftutilities.h>

double f(double x, double y) {return (x*x-y*y+1.0)/((x*x-y*y+1.0)*(x*x-y*y+1.0)+(2.0*x*y+1.0)*(2.0*x*y+1.0));};

/*!
  \example holomorphic.c
  In this example, we explore integration of a harmonic function:
  \f[
  f(x,y) = \frac{x^2-y^2+1}{(x^2-y^2+1)^2+(2xy+1)^2},
  \f]
  over the unit disk. In this case, we know from complex analysis that the integral of a holomorphic function is equal to \f$\pi\times f(0,0) = \pi/2\f$.

  More generally, the (weighted) integral follows from:
  \f[
  \int_0^{2\pi}\int_0^1 f(r\cos\theta, r\sin\theta) r^{2\alpha+1}(1-r^2)^\beta{\rm\,d}r{\rm\,d}\theta = \frac{\pi}{2}{\rm B}(\alpha+1,\beta+1),
  \f]
  where \f${\rm B}(x,y)\f$ is the beta function.

  We also know that the (weighted) square integral over the disk is related to the sum of the squares of the (generalized) Zernike coefficients. By:
  \f[
  \int_0^{2\pi}\int_0^1 [f(r\cos\theta, r\sin\theta)]^2r^{2\alpha+1}(1-r^2)^\beta{\rm\,d}r{\rm\,d}\theta = \pi\int_0^1\frac{r^{2\alpha+1}(1-r^2)^\beta}{2-r^4}{\rm\,d}r,
  \f]
  and:
  \f[
  \pi\int_0^1\frac{r^{2\alpha+1}(1-r^2)^\beta}{2-r^4}{\rm\,d}r = \frac{\pi^{\frac{3}{2}}}{2^{\alpha+\beta+3}}\frac{\Gamma(\alpha+1)\Gamma(\beta+1)}{\Gamma(\frac{\alpha+\beta+2}{2})\Gamma(\frac{\alpha+\beta+3}{2})} {}_3F_2\left(\begin{array}{c} 1, \frac{\alpha+1}{2}, \frac{\alpha+2}{2}\\ \frac{\alpha+\beta+2}{2}, \frac{\alpha+\beta+3}{2} \end{array}; \frac{1}{2}\right),
  \f]
  we may simplify the special case \f$(\alpha,\beta)=(0,0)\f$:
  \f[
  \int_0^{2\pi}\int_0^1 [f(r\cos\theta, r\sin\theta)]^2r{\rm\,d}r{\rm\,d}\theta = \frac{\pi}{2\sqrt{2}}\log(1+\sqrt{2}).
  \f]

  Provided we take \f$\alpha=0\f$ above, we may repeat the experiment with the Dunkl-Xu polynomials that rectangularize the disk.
*/
int main(void) {
    printf("In this example, we explore integration of a harmonic function over \n");
    printf("the unit disk. In this case, we know from complex analysis that the \n");
    printf("integral of a holomorphic function is equal to "MAGENTA("π × f(0,0)")".\n");
    printf("We analyze the function:\n");
    printf("\n");
    printf("\t"MAGENTA("f(x,y) = (x²-y²+1)/[(x²-y²+1)²+(2xy+1)²]")",\n");
    printf("\n");
    printf("on an "MAGENTA("N×M")" tensor product grid defined by:\n");
    printf("\n");
    printf("\t"MAGENTA("rₙ = cos[(n+1/2)π/2N]")", for "MAGENTA("0 ≤ n < N")",\n");
    printf("\n");
    printf("and\n");
    printf("\n");
    printf("\t"MAGENTA("θₘ = 2π m/M")", for "MAGENTA("0 ≤ m < M")";\n");
    printf("\n");
    printf("we convert the function samples to Chebyshev×Fourier coefficients using\n");
    printf(CYAN("ft_plan_disk_analysis")" and "CYAN("ft_execute_disk_analysis")"; and finally, we transform\n");
    printf("the Chebyshev×Fourier coefficients to disk harmonic coefficients using\n");
    printf(CYAN("ft_plan_disk2cxf")" and "CYAN("ft_execute_cxf2disk")".\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    char * FMT = "%1.3f";

    int N = 5;
    int M = 4*N-3;

    printf("\n\n"MAGENTA("N = %i")", and "MAGENTA("M = %i")"\n\n", N, M);

    double r[N], theta[M], F[4*N*N];

    for (int n = 0; n < N; n++)
        r[n] = sin((N-n-0.5)*M_PI/(2*N));
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

    double alpha = 0.0, beta = 0.0;
    ft_harmonic_plan * P = ft_plan_disk2cxf(N, alpha, beta);
    ft_disk_fftw_plan * PA = ft_plan_disk_analysis(N, M);

    ft_execute_disk_analysis('N', PA, F, N, M);
    ft_execute_cxf2disk('N', P, F, N, M);

    printf("Its Zernike coefficients are:\n\n");

    printmat("U", FMT, F, N, M);
    printf("\n");

    printf("The Zernike coefficients are useful for integration. The integral\n");
    printf("of "MAGENTA("f(x,y)")" over the disk should be "MAGENTA("π/2")" by harmonicity.\n");
    printf("The coefficient of "MAGENTA("Z_{0,0}")" multiplied by "MAGENTA("√π")" is: ");
    printf(FMT, F[0]*sqrt(M_PI));
    printf(".\n\n");
    printf("Using an orthonormal basis, the integral of "MAGENTA("[f(x,y)]^2")" over the disk is\n");
    printf("approximately the square of the 2-norm of the coefficients, ");
    printf(FMT, pow(ft_norm_1arg(F, N*M), 2));
    printf(".\n");
    printf("This compares favourably to the exact result, ");
    printf(FMT, M_PI/(2*sqrt(2))*log1p(sqrt(2)));
    printf(".\n\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_disk_fftw_plan(PA);

    printf("But there's more! Next, we repeat the experiment using the Dunkl-Xu\n");
    printf("orthonormal polynomials supported on the rectangularized disk.\n");

    N = 2*N;
    M = N;

    double w[N], x[N], z[M];

    for (int n = 0; n < N; n++) {
        w[n] = sin((n+0.5)*M_PI/N);
        x[n] = sin((N-2*n-1.0)*M_PI/(2*N));
    }
    for (int m = 0; m < M; m++)
        z[m] = sin((M-2*m-1.0)*M_PI/(2*M));

    printmat(MAGENTA("x")" grid", FMT, x, N, 1);
    printf("\n");
    printmat(MAGENTA("z")" grid", FMT, z, 1, M);
    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = f(x[n], w[n]*z[m]);

    printf("On the mapped tensor product grid, our function samples are:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    P = ft_plan_rectdisk2cheb(N, beta);
    ft_rectdisk_fftw_plan * QA = ft_plan_rectdisk_analysis(N, M);

    ft_execute_rectdisk_analysis('N', QA, F, N, M);
    ft_execute_cheb2rectdisk('N', P, F, N, M);

    printf("Its Dunkl-Xu coefficients are:\n\n");

    printmat("U", FMT, F, N, M);
    printf("\n");

    printf("The Dunkl-Xu coefficients are useful for integration. The integral\n");
    printf("of "MAGENTA("f(x,y)")" over the disk should be "MAGENTA("π/2")" by harmonicity.\n");
    printf("The coefficient of "MAGENTA("P_{0,0}")" multiplied by "MAGENTA("√π")" is: ");
    printf(FMT, F[0]*sqrt(M_PI));
    printf(".\n\n");
    printf("Using an orthonormal basis, the integral of "MAGENTA("[f(x,y)]^2")" over the disk is\n");
    printf("approximately the square of the 2-norm of the coefficients, ");
    printf(FMT, pow(ft_norm_1arg(F, N*M), 2));
    printf(".\n");
    printf("This compares favourably to the exact result, ");
    printf(FMT, M_PI/(2*sqrt(2))*log1p(sqrt(2)));
    printf(".\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_rectdisk_fftw_plan(QA);

    return 0;
}
