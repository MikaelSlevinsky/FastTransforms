#include "fasttransforms.h"
#include "ftutilities.h"

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "test_transforms_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "test_transforms_source.c"
#undef FLT
#undef X
#undef Y

#if defined(FT_QUADMATH)
    #define FLT long double
    #define X(name) FT_CONCAT(ft_, name, l)
    #define Y(name) FT_CONCAT(, name, l)
    #include "test_transforms_source.c"
    #undef FLT
    #undef X
    #undef Y
#endif

#include "test_transforms_mpfr.c"

int main(void) {
    int checksum = 0, n = 2048;
    printf("\nTesting methods for orthogonal polynomial transforms.\n");
    printf("\n\tSingle precision.\n\n");
    test_transformsf(&checksum, n);
    printf("\n\tDouble precision.\n\n");
    test_transforms(&checksum, n);
    printf("\n\tMulti-precision.\n\n");
    test_transforms_mpfr(&checksum, 256, 256, MPFR_RNDN);
    /*
    n = 16;
    printf("\nTesting methods for modified classical orthogonal polynomial transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    double alpha = 0.0, beta = 0.0;
    double u1[] = {0.9428090415820636, -0.32659863237109055, -0.42163702135578396, 0.2138089935299396}; // u1(x) = (1-x)^2*(1+x)
    ft_modified_plan * P = ft_plan_modified_jacobi_to_jacobi(n, alpha, beta, 4, u1, 0, NULL);
    double * DP1 = calloc(n*n, sizeof(double));
    double * IDP1 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP1[j+j*n] = DP1[j+j*n] = 1;
        ft_mpmv('N', P, DP1+j*n);
        ft_mpsv('N', P, IDP1+j*n);
    }
    ft_destroy_modified_plan(P);

    double u2[] = {0.9428090415820636, -0.32659863237109055, -0.42163702135578396, 0.2138089935299396}; // u2(x) = (1-x)^2*(1+x)
    double v2[] = {1.4142135623730951};
    P = ft_plan_modified_jacobi_to_jacobi(n, alpha, beta, 4, u2, 1, v2);
    double * DP2 = calloc(n*n, sizeof(double));
    double * IDP2 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP2[j+j*n] = DP2[j+j*n] = 1;
        ft_mpmv('N', P, DP2+j*n);
        ft_mpsv('N', P, IDP2+j*n);
    }
    ft_destroy_modified_plan(P);

    double err = ft_norm_2arg(DP1, DP2, n*n)/ft_norm_1arg(DP1, n*n);
    printf("Jacobi polynomial vs. trivial rational weight \t n = %3i |%20.2e ", n, err);
    ft_checktest(err, n*n, &checksum);

    free(DP1);
    free(IDP1);

    double u3[] = {-0.9428090415820636, 0.32659863237109055, 0.42163702135578396, -0.2138089935299396}; // u3(x) = -(1-x)^2*(1+x)
    double v3[] = {-5.185449728701348, 0.0, 0.42163702135578374}; // v3(x) = -(2-x)*(2+x)
    P = ft_plan_modified_jacobi_to_jacobi(n, alpha, beta, 4, u3, 3, v3);
    double * DP3 = calloc(n*n, sizeof(double));
    double * IDP3 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP3[j+j*n] = DP3[j+j*n] = 1;
        ft_mpmv('N', P, DP3+j*n);
        ft_mpsv('N', P, IDP3+j*n);
    }
    ft_destroy_modified_plan(P);

    alpha = 2.0;
    beta = 1.0;
    double u4[] = {1.1547005383792517};
    double v4[] = {4.387862045841156, 0.1319657758147716, -0.20865621238292037}; // v4(x) = (2-x)*(2+x)
    P = ft_plan_modified_jacobi_to_jacobi(n, alpha, beta, 1, u4, 3, v4);
    double * DP4 = calloc(n*n, sizeof(double));
    double * IDP4 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP4[j+j*n] = DP4[j+j*n] = 1;
        ft_mpmv('N', P, DP4+j*n);
        ft_mpsv('N', P, IDP4+j*n);
    }
    ft_destroy_modified_plan(P);

    ft_trmm('N', n, DP2, n, DP4, n, n);

    err = ft_norm_2arg(DP3, DP4, n*n)/ft_norm_1arg(DP4, n*n);
    printf("Jacobi rational vs. raised polynomial weight \t n = %3i |%20.2e ", n, err);
    ft_checktest(err, pow(n+1, 2), &checksum);

    free(DP2);
    free(DP3);
    free(DP4);
    free(IDP2);
    free(IDP3);
    free(IDP4);

    alpha = 0.0;
    double u5[] = {2.0, -4.0, 2.0}; // u5(x) = x^2
    P = ft_plan_modified_laguerre_to_laguerre(n, alpha, 3, u5, 0, NULL);
    double * DP5 = calloc(n*n, sizeof(double));
    double * IDP5 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP5[j+j*n] = DP5[j+j*n] = 1;
        ft_mpmv('N', P, DP5+j*n);
        ft_mpsv('N', P, IDP5+j*n);
    }
    ft_destroy_modified_plan(P);

    double u6[] = {2.0, -4.0, 2.0}; // u6(x) = x^2
    double v6[] = {1.0};
    P = ft_plan_modified_laguerre_to_laguerre(n, alpha, 3, u6, 1, v6);
    double * DP6 = calloc(n*n, sizeof(double));
    double * IDP6 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP6[j+j*n] = DP6[j+j*n] = 1;
        ft_mpmv('N', P, DP6+j*n);
        ft_mpsv('N', P, IDP6+j*n);
    }
    ft_destroy_modified_plan(P);

    double u7[] = {2.0, -4.0, 2.0}; // u7(x) = x^2
    double v7[] = {7.0, -7.0, 2.0}; // v7(x) = (1+x)*(2+x)
    P = ft_plan_modified_laguerre_to_laguerre(n, alpha, 3, u7, 3, v7);
    double * DP7 = calloc(n*n, sizeof(double));
    double * IDP7 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP7[j+j*n] = DP7[j+j*n] = 1;
        ft_mpmv('N', P, DP7+j*n);
        ft_mpsv('N', P, IDP7+j*n);
    }
    ft_destroy_modified_plan(P);

    alpha = 2.0;
    double u8[] = {sqrt(2.0)};
    double v8[] = {sqrt(1058.0), -sqrt(726.0), sqrt(48.0)}; // v8(x) = (1+x)*(2+x)
    P = ft_plan_modified_laguerre_to_laguerre(n, alpha, 1, u8, 3, v8);
    double * DP8 = calloc(n*n, sizeof(double));
    double * IDP8 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP8[j+j*n] = DP8[j+j*n] = 1;
        ft_mpmv('N', P, DP8+j*n);
        ft_mpsv('N', P, IDP8+j*n);
    }
    ft_destroy_modified_plan(P);

    err = ft_norm_2arg(DP5, DP6, n*n)/ft_norm_1arg(DP5, n*n);
    printf("Laguerre polynomial vs. trivial rational weight\t n = %3i |%20.2e ", n, err);
    ft_checktest(err, n*n, &checksum);

    ft_trmm('N', n, DP6, n, DP8, n, n);

    err = ft_norm_2arg(DP7, DP8, n*n)/ft_norm_1arg(DP8, n*n);
    printf("Laguerre rational vs. raised polynomial weight \t n = %3i |%20.2e ", n, err);
    ft_checktest(err, pow(n+1, 4), &checksum);

    free(DP5);
    free(DP6);
    free(DP7);
    free(DP8);
    free(IDP5);
    free(IDP6);
    free(IDP7);
    free(IDP8);

    double u9[] = {2.995504568550877, 0.0, 3.7655850551068593, 0.0, 1.6305461589167827};
    double v9[] = {2.995504568550877, 0.0, 3.7655850551068593, 0.0, 1.6305461589167827};
    P = ft_plan_modified_hermite_to_hermite(n, 5, u9, 5, v9);
    double * DP9 = calloc(n*n, sizeof(double));
    double * IDP9 = calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++) {
        IDP9[j+j*n] = DP9[j+j*n] = 1;
        ft_mpmv('N', P, DP9+j*n);
        ft_mpsv('N', P, IDP9+j*n);
    }
    ft_destroy_modified_plan(P);

    err = ft_norm_2arg(DP9, IDP9, n*n)/ft_norm_1arg(DP9, n*n);
    printf("Hermite trivial rational weight \t\t n = %3i |%20.2e ", n, err);
    ft_checktest(err, pow(n+1, 3), &checksum);

    free(DP9);
    free(IDP9);
    */
    n = 128;
    printf("\nTesting methods for associated classical orthogonal polynomial transforms.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int c = 1; c < 9; c++) {
        double alpha = 0.123, beta = 0.456, gamma = 0.789, delta = 1.23, err = 0;
        for (int norm1 = 0; norm1 <= 1; norm1++) {
            for (int norm2 = 0; norm2 <= 1; norm2++) {
                double * V = plan_associated_jacobi_to_jacobi(norm1, norm2, n, c, alpha, beta, gamma, delta);
                ft_banded * M = ft_create_jacobi_multiplication(norm2, n, n, gamma, delta);
                double * x = malloc(n*sizeof(double));
                for (int nu = 1; nu < n-1; nu++) {
                    for (int i = 0; i < n; i++)
                        x[i] = ft_rec_B_jacobi(norm1, nu+c, alpha, beta)*V[i+nu*n] - ft_rec_C_jacobi(norm1, nu+c, alpha, beta)*V[i+(nu-1)*n];
                    ft_gbmv(ft_rec_A_jacobi(norm1, nu+c, alpha, beta), M, V+nu*n, 1, x);
                    err += ft_norm_2arg(V+(nu+1)*n, x, n)/ft_norm_1arg(x, n);
                }
                ft_destroy_banded(M);
                free(x);
                free(V);
            }
        }
        printf("Associated Jacobi recurrence \t (n, c) = (%3i, %i) \t |%20.2e ", n, c, err);
        ft_checktest(err, n*n, &checksum);
    }
    for (int c = 1; c < 9; c++) {
        double alpha = 0.123, beta = 0.456, err = 0;
        for (int norm1 = 0; norm1 <= 1; norm1++) {
            for (int norm2 = 0; norm2 <= 1; norm2++) {
                double * V = plan_associated_laguerre_to_laguerre(norm1, norm2, n, c, alpha, beta);
                ft_banded * M = ft_create_laguerre_multiplication(norm2, n, n, beta);
                double * x = malloc(n*sizeof(double));
                for (int nu = 1; nu < n-1; nu++) {
                    for (int i = 0; i < n; i++)
                        x[i] = ft_rec_B_laguerre(norm1, nu+c, alpha)*V[i+nu*n] - ft_rec_C_laguerre(norm1, nu+c, alpha)*V[i+(nu-1)*n];
                    ft_gbmv(ft_rec_A_laguerre(norm1, nu+c, alpha), M, V+nu*n, 1, x);
                    err += ft_norm_2arg(V+(nu+1)*n, x, n)/ft_norm_1arg(x, n);
                }
                ft_destroy_banded(M);
                free(x);
                free(V);
            }
        }
        printf("Associated Laguerre recurrence \t (n, c) = (%3i, %i) \t |%20.2e ", n, c, err);
        ft_checktest(err, n*n, &checksum);
    }
    for (int c = 1; c < 9; c++) {
        double err = 0;
        for (int norm1 = 0; norm1 <= 1; norm1++) {
            for (int norm2 = 0; norm2 <= 1; norm2++) {
                double * V = plan_associated_hermite_to_hermite(norm1, norm2, n, c);
                ft_banded * M = ft_create_hermite_multiplication(norm2, n, n);
                double * x = malloc(n*sizeof(double));
                for (int nu = 1; nu < n-1; nu++) {
                    for (int i = 0; i < n; i++)
                        x[i] = ft_rec_B_hermite(norm1, nu+c)*V[i+nu*n] - ft_rec_C_hermite(norm1, nu+c)*V[i+(nu-1)*n];
                    ft_gbmv(ft_rec_A_hermite(norm1, nu+c), M, V+nu*n, 1, x);
                    err += ft_norm_2arg(V+(nu+1)*n, x, n)/ft_norm_1arg(x, n);
                }
                ft_destroy_banded(M);
                free(x);
                free(V);
            }
        }
        printf("Associated Hermite recurrence \t (n, c) = (%3i, %i) \t |%20.2e ", n, c, err);
        ft_checktest(err, n*n, &checksum);
    }
    for (int c = 1; c < 9; c++) {
        double * V = plan_associated_jacobi_to_jacobi(0, 0, n, c, 0.0, 0.0, 0.0, 0.0);
        double * Pnc1 = malloc(n*sizeof(double));
        for (int i = 0; i < n; i++)
            Pnc1[i] = ((double) c)/(c+i);
        for (int i = 1; i < n; i++)
            Pnc1[i] += Pnc1[i-1];
        double * colsum = calloc(n, sizeof(double));
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                colsum[j] += V[i+j*n];
        double err = ft_norm_2arg(Pnc1, colsum, n)/ft_norm_1arg(Pnc1, n);
        printf("Error in evaluating P_n(1; %i) \t\t\t\t |%20.2e ", c, err);
        ft_checktest(err, 4, &checksum);
        free(V);
        free(Pnc1);
        free(colsum);
    }
    printf("\n");
    return checksum;
}
