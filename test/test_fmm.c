#include "fasttransforms.h"
#include "ftutilities.h"
#include "fmm.h"

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#define FT_USE_DOUBLE
#include "test_fmm_source.c"
#undef FLT
#undef X
#undef Y
#undef FT_USE_DOUBLE

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#define FT_USE_SINGLE
#include "test_fmm_source.c"
#undef FLT
#undef X
#undef Y
#undef FT_USE_SINGLE

int main(void) {
    int checksum = 0;
    printf("\nTesting fast multipole method for Legendre-Chebyshev transforms.\n");
    printf("\n\tSingle precision.\n\n");
    test_fmm_transformsf(&checksum, 1e5);
    test_fmm_transforms2df(&checksum, 2048);
    printf("\n\tDouble precision.\n\n");
    test_fmm_transforms(&checksum, 1e5);
    test_fmm_transforms2d(&checksum, 2048);
    //test_fmm(8192, 64, 18, 0.0, 1, 0, 2);
    //test_fmmf(8192, 32, 8, 0.0, 1, 0, 2);
    //test_fmm_speed(8192, 64, 1000, L2C, 18, 0, 2);
    //test_fmm_speedf(8192, 32, 1000, L2C, 8, 0, 2);
    return 0;
}
