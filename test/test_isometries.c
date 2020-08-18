#include "fasttransforms.h"
#include "ftinternal.h"
#include "ftutilities.h"
#include "isometries.h"

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "test_isometries_source.c"
#undef FLT
#undef X
#undef Y

int main(void){
	int checksum = 0;
	printf("\nTesting methods for isometries.\n");
	printf("\n\tDouble Precision.\n\n");
	test_isometries(&checksum);

	return checksum;
}
