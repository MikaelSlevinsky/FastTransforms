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
    printf("\nTesting methods for classical orthogonal polynomial transforms.\n");
    printf("\n\tSingle precision.\n\n");
    test_transformsf(&checksum, n);
    printf("\n\tDouble precision.\n\n");
    test_transforms(&checksum, n);
    printf("\n\tMulti-precision.\n\n");
    test_transforms_mpfr(&checksum, 256, 256, MPFR_RNDN);
    n = 32;
    printf("\nTesting methods for modified classical orthogonal polynomial transforms.\n");
    printf("\n\tSingle precision.\n\n");
    test_modified_transformsf(&checksum, n);
    printf("\n\tDouble precision.\n\n");
    test_modified_transforms(&checksum, n);
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
