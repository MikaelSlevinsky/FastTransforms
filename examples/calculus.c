#include <fasttransforms.h>
#include <ftutilities.h>

double f(double x, double y) {return 1.0/(1.0+x*x+y*y);};
double fx(double x, double y) {return -2.0*x/(1.0+x*x+y*y)/(1.0+x*x+y*y);};
double fy(double x, double y) {return -2.0*y/(1.0+x*x+y*y)/(1.0+x*x+y*y);};

/*!
  \example calculus.c
  In this example, we sample a bivariate function:
  \f[
  f(x,y) = \frac{1}{1+x^2+y^2},
  \f]
  on the reference triangle with vertices \f$(0,0)\f$, \f$(0,1)\f$, and \f$(1,0)\f$ and analyze it in a Proriol series. Then, we find Proriol series for each component of its gradient by term-by-term differentiation of our expansion, and we compare them with the "true" Proriol series by sampling an exact expression for the gradient.
*/
int main(void) {
    printf("In this example, we sample a bivariate function "MAGENTA("f(x,y)")" on the reference triangle\n");
    printf("with vertices "MAGENTA("(0,0)")", "MAGENTA("(0,1)")", and "MAGENTA("(1,0)")" and analyze it in a Proriol series.\n");
    printf("Then, we find Proriol series for each component of its gradient by term-by-term\n");
    printf("differentiation of our expansion, and we compare them with the \"true\" Proriol\n");
    printf("series by sampling an exact expression for the gradient.\n");
    printf("\n");
    printf("We analyze the function:\n");
    printf("\n");
    printf("\t"MAGENTA("f(x,y) = 1/(1+x²+y²)")",\n");
    printf("\n");
    printf("on an "MAGENTA("N×M")" mapped tensor product grid defined by:\n");
    printf("\n");
    printf("\t"MAGENTA("x = (1+u)/2")", and "MAGENTA("y = (1-u)*(1+v)/4")", where:\n");
    printf("\n");
    printf("\t"MAGENTA("uₙ = cos[(n+1/2)π/N]")", for "MAGENTA("0 ≤ n < N")", and\n");
    printf("\n");
    printf("\t"MAGENTA("vₘ = cos[(m+1/2)π/M]")", for "MAGENTA("0 ≤ m < M")";\n");
    printf("\n");
    printf("we convert the function samples to mapped Chebyshev² coefficients using\n");
    printf(CYAN("ft_plan_tri_analysis")" and "CYAN("ft_execute_tri_analysis")"; and finally, we transform\n");
    printf("the mapped Chebyshev² coefficients to Proriol coefficients using\n");
    printf(CYAN("ft_plan_tri2cheb")" and "CYAN("ft_execute_cheb2tri")".\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    char * FMT = "%1.3f";

    ft_harmonic_plan * P, * Px, * Py;
    ft_triangle_fftw_plan * PA;

    int N = 10;
    int M = N;

    printf("\n\n"MAGENTA("N = %i")", and "MAGENTA("M = %i")"\n\n", N, M);

    double alpha = 0.0, beta = 0.0, gamma = 0.0;

    printf("The Proriol series has parameters: "MAGENTA("(α,β,γ) = (%1.3f,%1.3f,%1.3f)")".\n\n", alpha, beta, gamma);

    P = ft_plan_tri2cheb(N, alpha, beta, gamma);
    Px = ft_plan_tri2cheb(N, alpha+1.0, beta, gamma+1.0);
    Py = ft_plan_tri2cheb(N, alpha, beta+1.0, gamma+1.0);
    PA = ft_plan_tri_analysis(N, M);

    double u[N], x[N], v[M], w[M], F[N*M], Fx[N*M], Fy[N*M], Gx[N*M], Gy[N*M];

    for (int n = 0; n < N; n++) {
        u[n] = sin((N-2.0*n-1.0)*M_PI/(2*N));
        x[n] = pow(sin((2.0*N-2.0*n-1.0)*M_PI/(4*N)), 2);
    }
    for (int m = 0; m < M; m++) {
        v[m] = sin((M-2.0*m-1.0)*M_PI/(2*M));
        w[m] = pow(sin((2.0*M-2.0*m-1.0)*M_PI/(4*M)), 2);
    }

    printmat(MAGENTA("u")" grid", FMT, u, N, 1);
    printf("\n");
    printmat(MAGENTA("v")" grid", FMT, v, 1, M);
    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = f(x[n], x[N-1-n]*w[m]);

    printf("On the mapped tensor product grid, our function samples are:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_tri_analysis(PA, F, N, M);
    ft_execute_cheb2tri(P, F, N, M);

    printf("Its Proriol-"MAGENTA("(α,β,γ)")" coefficients are:\n\n");

    printmat("U", FMT, F, N, M);
    printf("\n");

    printf("Similarly, our function's gradient samples are:\n\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            Fx[n+N*m] = fx(x[n], x[N-1-n]*w[m]);

    printmat("Fx", FMT, Fx, N, M);
    printf("\n");

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            Fy[n+N*m] = fy(x[n], x[N-1-n]*w[m]);

    printmat("Fy", FMT, Fy, N, M);
    printf("\n");

    printf("For the partial derivative with respect to "MAGENTA("x")", Olver et al.\n");
    printf("derive simple expressions for the representation of this component\n");
    printf("using a Proriol-"MAGENTA("(α+1,β,γ+1)")" series. For the partial derivative with respect\n");
    printf("to "MAGENTA("y")", the analogous formulae result in a Proriol-"MAGENTA("(α,β+1,γ+1)")" series.\n");
    printf("These expressions are adapted from "CYAN("https://doi.org/10.1137/19M1245888")"\n");

    ft_execute_tri_analysis(PA, Fx, N, M);
    ft_execute_cheb2tri(Px, Fx, N, M);

    printmat("Ux from sampling", FMT, Fx, N, M);
    printf("\n");

    double cf1, cf2;

    for (int n = 0; n < N-1; n++)
        Gx[n+N*(M-1)] = 0.0;
    for (int m = 0; m < M; m++)
        Gx[N-1+N*m] = 0.0;
    for (int m = 0; m < M-1; m++) {
        for (int n = 0; n < N-1; n++) {
            cf1 = sqrt((n+1.0)*(n+2.0*m+alpha+beta+gamma+3.0)/(2.0*m+beta+gamma+1.0)*(m+beta+gamma+1.0)/(2.0*m+beta+gamma+2.0)*(m+gamma+1.0)*8.0);
            cf2 = sqrt((n+alpha+1.0)*(m+1.0)/(2.0*m+beta+gamma+2.0)*(m+beta+1.0)/(2.0*m+beta+gamma+3.0)*(n+2.0*m+beta+gamma+3.0)*8.0);
            Gx[n+N*m] = cf1*F[n+1+N*m] + cf2*F[n+N*(m+1)];
        }
    }

    printmat("Ux from U", FMT, Gx, N, M);
    printf("\n");

    ft_execute_tri_analysis(PA, Fy, N, M);
    ft_execute_cheb2tri(Py, Fy, N, M);

    printmat("Uy from sampling", FMT, Fy, N, M);
    printf("\n");

    for (int n = 0; n < N-1; n++)
        Gy[n+N*(M-1)] = 0.0;
    for (int m = 0; m < M; m++)
        Gy[N-1+N*m] = 0.0;
    for (int m = 0; m < M-1; m++)
        for (int n = 0; n < N-1; n++)
            Gy[n+N*m] = 4.0*sqrt((m+1.0)*(m+beta+gamma+2.0))*F[n+N*(m+1)];

    printmat("Uy from U", FMT, Gy, N, M);
    printf("\n");

    printf("The 2-norm relative error in differentiating the Proriol series\n");
    printf("for "MAGENTA("f(x,y)")" term-by-term and its sampled gradient is %4.2e.\n", sqrt(pow(ft_norm_2arg(Fx, Gx, N*M), 2) + pow(ft_norm_2arg(Fy, Gy, N*M), 2))/sqrt(pow(ft_norm_1arg(Fx, N*M), 2) + pow(ft_norm_1arg(Fy, N*M), 2)));
    printf("This error can be improved upon by increasing "MAGENTA("N")" and "MAGENTA("M")".\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_harmonic_plan(Px);
    ft_destroy_harmonic_plan(Py);
    ft_destroy_triangle_fftw_plan(PA);

    return 0;
}
