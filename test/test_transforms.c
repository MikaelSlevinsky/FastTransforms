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
    printf("\n\tAssociated orthogonal polynomial transforms.\n\n");
    for (int c = 1; c < 9; c++) {
        n = 128;
        double * V = plan_associated_jacobi_to_jacobi(0, n, c, 0.0, 0.0, 0.0, 0.0);
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
