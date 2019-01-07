#include "fasttransforms.h"
#include "ftutilities.h"

void test_arrowf(int * checksum);
void test_arrow (int * checksum);
void test_arrowl(int * checksum);
void test_arrowq(int * checksum);

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for bidiagonal and symmetric arrow matrices.\n");
    printf("\n\tSingle precision.\n\n");
    test_arrowf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_arrow(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_arrowl(&checksum);
    printf("\n\tQuadruple precision.\n\n");
    test_arrowq(&checksum);
    printf("\n");
    return checksum;
}

#define FLT float
#define X(name) CONCAT(, name, f)
#include "test_arrow_source.c"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "test_arrow_source.c"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "test_arrow_source.c"
#undef FLT
#undef X

#define FLT quadruple
#define X(name) CONCAT(, name, q)
#include "test_arrow_source.c"
#undef FLT
#undef X
