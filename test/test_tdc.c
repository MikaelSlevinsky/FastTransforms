#include "fasttransforms.h"
#include "ftutilities.h"

void test_tdcf(int * checksum);
void test_tdc (int * checksum);
void test_tdcl(int * checksum);
void test_tdcq(int * checksum);

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for symmetric-definite tridiagonal divide and conquer.\n");
    printf("\n\tSingle precision.\n\n");
    test_tdcf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_tdc(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_tdcl(&checksum);
    printf("\n\tQuadruple precision.\n\n");
    test_tdcq(&checksum);
    printf("\n");
    return checksum;
}

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "test_tdc_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "test_tdc_source.c"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "test_tdc_source.c"
#undef FLT
#undef X
#undef Y

#define FLT quadruple
#define X(name) FT_CONCAT(ft_, name, q)
#define Y(name) FT_CONCAT(, name, q)
#include "test_tdc_source.c"
#undef FLT
#undef X
#undef Y
