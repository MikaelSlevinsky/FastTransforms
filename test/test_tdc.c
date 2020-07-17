#include "fasttransforms.h"
#include "ftutilities.h"

#if defined(FT_QUADMATH)
    #define FLT quadruple
    #define X(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, q)
    #include "test_tdc_source.c"
    #undef FLT
    #undef X
    #undef Y
#endif

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "test_tdc_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define X2(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, )
#include "test_tdc_source.c"
#include "test_tdc_source2.c"
#undef FLT
#undef X
#undef X2
#undef Y

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define X2(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, f)
#include "test_tdc_source.c"
#include "test_tdc_source2.c"
#undef FLT
#undef X
#undef X2
#undef Y

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for symmetric-definite tridiagonal divide and conquer.\n");
    printf("\n\tSingle precision.\n\n");
    test_tdcf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_tdc(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_tdcl(&checksum);
    #if defined(FT_QUADMATH)
        printf("\n\tQuadruple precision.\n\n");
        test_tdcq(&checksum);
    #endif
    printf("\n");
    printf("\nTesting methods for dropping the precision.\n");
    printf("\n\tDouble ↘  single precision.\n\n");
    test_tdc_drop_precisionf(&checksum);
    printf("\n\tLong double ↘  double precision.\n\n");
    test_tdc_drop_precision(&checksum);
    printf("\n");
    return 0;
}
