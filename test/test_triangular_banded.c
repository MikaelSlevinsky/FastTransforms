#include "fasttransforms.h"
#include "ftutilities.h"

void test_triangular_bandedf(int * checksum);
void test_triangular_banded (int * checksum);
void test_triangular_bandedl(int * checksum);
void test_triangular_bandedq(int * checksum);

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for triangular banded matrices.\n");
    printf("\n\tSingle precision.\n\n");
    //test_triangular_bandedf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_triangular_banded(&checksum);
    printf("\n\tLong double precision.\n\n");
    //test_triangular_bandedl(&checksum);
    printf("\n\tQuadruple precision.\n\n");
    //test_triangular_bandedq(&checksum);
    printf("\n");
    return checksum;
}

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "test_triangular_banded_source.c"
#undef FLT
#undef X
#undef Y
