#include "fasttransforms.h"
#include "ftutilities.h"

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "test_banded_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "test_banded_source.c"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "test_banded_source.c"
#undef FLT
#undef X
#undef Y

#if defined(FT_QUADMATH)
    #define FLT quadruple
    #define X(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, q)
    #include "test_banded_source.c"
    #undef FLT
    #undef X
    #undef Y
#endif

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for banded matrices.\n");
    printf("\n\tSingle precision.\n\n");
    test_bandedf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_banded(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_bandedl(&checksum);
    #if defined(FT_QUADMATH)
        printf("\n\tQuadruple precision.\n\n");
        test_bandedq(&checksum);
    #endif
    printf("\n");
    return checksum;
}
