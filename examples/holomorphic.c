#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fasttransforms.h"

void printmat(char * MAT, char * FMT, double * A, int n, int m);
double f(double x, double y);
double vecnorm_1arg(double * A, int n, int m);

int main(void) {
    printf("In this example, we explore integration of a harmonic function over \n");
    printf("the unit disk. In this case, we know from complex analysis that the \n");
    printf("integral of a holomorphic function is equal to π × f(0,0).\n");
    printf("We analyze the function:\n");
    printf("\n");
    printf("\tf(x,y) = (x²-y²+1)/[(x²-y²+1)²+(2xy+1)²],\n");
    printf("\n");
    printf("on an N×M tensor product grid defined by:\n");
    printf("\n");
    printf("\trₙ = cos[(n+1/2)π/2N], for 0 ≤ n < N,\n");
    printf("\n");
    printf("and\n");
    printf("\n");
    printf("\tφₘ = 2π m/M, for 0 ≤ m < M;\n");
    printf("\n");
    printf("we convert the function samples to Chebyshev×Fourier coefficients using\n");
    printf("`ft_plan_disk_analysis` and `ft_execute_disk_analysis`; and finally, we transform\n");
    printf("the Chebyshev×Fourier coefficients to disk harmonic coefficients using\n");
    printf("`ft_plan_disk2cxf` and `ft_execute_cxf2disk`.\n");
    printf("\n");
    printf("N.B. for the storage pattern of the printed arrays, please consult the\n");
    printf("documentation. (Arrays are stored in column-major ordering.)\n");

    char * FMT = "%1.3f";

    ft_harmonic_plan * P;
    ft_disk_fftw_plan * PA;

    int N = 5;
    int M = 4*N-3;

    printf("\n\nN = %i, and M = %i\n\n", N, M);

    P = ft_plan_disk2cxf(N);
    PA = ft_plan_disk_analysis(N, M);

    double * r = malloc(N*sizeof(double));
    double * theta = malloc(M*sizeof(double));

    for (int n = 0; n < N; n++)
        r[n] = cos((n+0.5)*M_PI/(2*N));
    for (int m = 0; m < M; m++)
        theta[m] = 2.0*M_PI*m/M;

    printmat("Radial grid r", FMT, r, N, 1);
    printf("\n");
    printmat("Azimuthal grid θ", FMT, theta, 1, M);
    printf("\n");

    double * F = calloc(N*M, sizeof(double));

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            F[n+N*m] = f(r[n]*cos(theta[m]), r[n]*sin(theta[m]));

    printf("On the tensor product grid, our function samples are:\n\n");

    printmat("F", FMT, F, N, M);
    printf("\n");

    ft_execute_disk_analysis(PA, F, N, M);
    ft_execute_cxf2disk(P, F, N, M);

    printf("Its Zernike coefficients are:\n\n");

    printmat("U", FMT, F, N, M);
    printf("\n");

    printf("The Zernike coefficients are useful for integration. The integral\n");
    printf("of f(x,y) over the disk should be π/2 by harmonicity.\n");
    printf("The coefficient of Z_0^0 multiplied by √π is: ");
    printf(FMT, F[0]*sqrt(M_PI));
    printf(".\n\n");
    printf("Using an orthonormal basis, the integral of [f(x,y)]^2 over the\n");
    printf("disk is approximately the square of the 2-norm of the coefficients, ");
    printf(FMT, pow(vecnorm_1arg(F, N, M), 2));
    printf(".\n");

    ft_destroy_harmonic_plan(P);
    ft_destroy_disk_fftw_plan(PA);
    free(r);
    free(theta);
    free(F);

    return 0;
}

#define A(i,j) A[(i)+n*(j)]

void printmat(char * MAT, char * FMT, double * A, int n, int m) {
    printf("%s = \n", MAT);
    if (n > 0 && m > 0) {
        if (A(0,0) < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, A(0,0));
        for (int j = 1; j < m; j++) {
            if (A(0,j) < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, A(0,j));
        }
        for (int i = 1; i < n-1; i++) {
            printf("\n");
            if (A(i,0) < 0) {printf(" ");}
            else {printf("  ");}
            printf(FMT, A(i,0));
            for (int j = 1; j < m; j++) {
                if (A(i,j) < 0) {printf("  ");}
                else {printf("   ");}
                printf(FMT, A(i,j));
            }
        }
        if (n > 1) {
            printf("\n");
            if (A(n-1,0) < 0) {printf(" ");}
            else {printf("  ");}
            printf(FMT, A(n-1,0));
            for (int j = 1; j < m; j++) {
                if (A(n-1,j) < 0) {printf("  ");}
                else {printf("   ");}
                printf(FMT, A(n-1,j));
            }
        }
        printf("]\n");
    }
}

double f(double x, double y) {return (x*x-y*y+1.0)/((x*x-y*y+1.0)*(x*x-y*y+1.0)+(2.0*x*y+1.0)*(2.0*x*y+1.0));};

double vecnorm_1arg(double * A, int n, int m) {
    double ret = 0.0;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            ret += pow(A(i,j), 2);
    return sqrt(ret);
}
