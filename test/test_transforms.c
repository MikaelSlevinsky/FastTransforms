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

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "test_transforms_source.c"
#undef FLT
#undef X
#undef Y

#ifdef FT_USE_MPFR
    #include "test_transforms_mpfr.c"
#endif

int main(void) {
    int checksum = 0, n = 2048;
    printf("\nTesting methods for orthogonal polynomial transforms.\n");
    printf("\n\tSingle precision.\n\n");
    test_transformsf(&checksum, n);
    printf("\n\tDouble precision.\n\n");
    test_transforms(&checksum, n);
    #ifdef FT_USE_MPFR
        printf("\n\tMulti-precision.\n\n");
        test_transforms_mpfr(&checksum, 256, 256, MPFR_RNDN);
    #endif
    printf("\n");
    return checksum;
}
