#include "fasttransforms.h"
#include "ftutilities.h"

void test_hierarchicalf(int * checksum);
void test_hierarchical (int * checksum);
void test_hierarchicall(int * checksum);
void test_hierarchicalq(int * checksum);

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for hierarchical matrices.\n");
    printf("\n\tSingle precision.\n\n");
    test_hierarchicalf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_hierarchical(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_hierarchicall(&checksum);
    printf("\n\tQuadruple precision.\n\n");
    test_hierarchicalq(&checksum);
    printf("\n");
    return checksum;
}

#define FLT float
#define X(name) CONCAT(, name, f)
#include "test_hierarchical_source.c"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "test_hierarchical_source.c"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "test_hierarchical_source.c"
#undef FLT
#undef X

#define FLT quadruple
#define X(name) CONCAT(, name, q)
#include "test_hierarchical_source.c"
#undef FLT
#undef X
