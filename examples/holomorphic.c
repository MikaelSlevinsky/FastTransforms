#include "fasttransforms.h"
#include "ftutilities.h"

double f(double x, double y) {return (x*x-y*y+1.0)/((x*x-y*y+1.0)*(x*x-y*y+1.0)+(2.0*x*y+1.0)*(2.0*x*y+1.0));};

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

    double r[N], theta[M], F[N*M];

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

    ft_harmonic_plan * P = ft_plan_disk2cxf(N);
    ft_disk_fftw_plan * PA = ft_plan_disk_analysis(N, M);

    ft_execute_disk_analysis(PA, F, N, M);
    ft_execute_cxf2disk(P, F, N, M);

    printf("Its Zernike coefficients are:\n\n");

    printmat("U", FMT, F, N, M);
    printf("\n");

    printf("The Zernike coefficients are useful for integration. The integral\n");
    printf("of "MAGENTA("f(x,y)")" over the disk should be "MAGENTA("π/2")" by harmonicity.\n");
    printf("The coefficient of "MAGENTA("Z_0^0")" multiplied by "MAGENTA("√π")" is: ");
    printf(FMT, F[0]*sqrt(M_PI));
    printf(".\n\n");
    printf("Using an orthonormal basis, the integral of "MAGENTA("[f(x,y)]^2")" over the\n");
    printf("disk is approximately the square of the 2-norm of the coefficients, ");
    printf(FMT, pow(norm_1arg(F, N*M), 2));
    printf(".\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_disk_fftw_plan(PA);

    return 0;
}
