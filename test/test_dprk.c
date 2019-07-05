#include "fasttransforms.h"
#include "ftutilities.h"

void test_dprkf(int * checksum);
void test_dprk (int * checksum);
void test_dprkl(int * checksum);
void test_dprkq(int * checksum);

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for symmetric diagonal-plus-rank-k matrices.\n");
    printf("\n\tSingle precision.\n\n");
    test_dprkf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_dprk(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_dprkl(&checksum);
    printf("\n\tQuadruple precision.\n\n");
    test_dprkq(&checksum);
    printf("\n");
    return checksum;
}

#define FLT float
#define X(name) CONCAT(, name, f)
#include "test_dprk_source.c"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "test_dprk_source.c"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "test_dprk_source.c"
#undef FLT
#undef X

#define FLT quadruple
#define X(name) CONCAT(, name, q)
#include "test_dprk_source.c"
#undef FLT
#undef X
