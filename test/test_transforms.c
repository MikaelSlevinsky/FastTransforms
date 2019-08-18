#include "fasttransforms.h"
#include "ftutilities.h"

void test_transformsf(int * checksum, int n);
void test_transforms (int * checksum, int n);
void test_transformsl(int * checksum, int n);

int main(void) {
    int checksum = 0, n = 2048;
    printf("\nTesting methods for orthogonal polynomial transforms.\n");
    printf("\n\tSingle precision.\n\n");
    test_transformsf(&checksum, n);
    printf("\n\tDouble precision.\n\n");
    test_transforms(&checksum, n);
    printf("\n\tLong double precision.\n\n");
    test_transformsl(&checksum, n>>1);
    printf("\n");
    return checksum;
}

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
