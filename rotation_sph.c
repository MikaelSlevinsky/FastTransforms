#include "src/ftutilities.h"
#include "src/fasttransforms.h"
#include <cblas.h>
#include <math.h>
#include <openblas/lapack.h>

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "rotation_sph_source.c"
#undef FLT
#undef X
#undef Y

int main(){
	ft_do_a_test();
		
	return 0;
}
