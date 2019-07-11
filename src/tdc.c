#include "fasttransforms.h"
#include "ftinternal.h"

#define FLT quadruple
#define X(name) FT_CONCAT(ft_, name, q)
#define Y(name) FT_CONCAT(, name, q)
#define BLOCKRANK 2*((int) floor(-log(Y(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "dprk_source.c"
#include "tridiagonal_source.c"
#include "triangular_banded_source.c"
#include "hierarchical_source.c"
#include "tdc_source.c"
#undef FLT
#undef X
#undef Y
#undef BLOCKRANK
#undef BLOCKSIZE

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#define BLOCKRANK 2*((int) floor(-log(Y(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "dprk_source.c"
#include "tridiagonal_source.c"
#include "triangular_banded_source.c"
#include "hierarchical_source.c"
#include "tdc_source.c"
#undef FLT
#undef X
#undef Y
#undef BLOCKRANK
#undef BLOCKSIZE

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define X2(name) FT_CONCAT(ft_, name, q)
#define Y(name) FT_CONCAT(, name, )
#define BLOCKRANK 2*((int) floor(-log(Y(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "dprk_source.c"
#include "tridiagonal_source.c"
#include "triangular_banded_source.c"
#include "hierarchical_source.c"
#include "tdc_source.c"
#include "tdc_source2.c"
#undef FLT
#undef X
#undef X2
#undef Y
#undef BLOCKRANK
#undef BLOCKSIZE

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define X2(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, f)
#define BLOCKRANK 2*((int) floor(-log(Y(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "dprk_source.c"
#include "tridiagonal_source.c"
#include "triangular_banded_source.c"
#include "hierarchical_source.c"
#include "tdc_source.c"
#include "tdc_source2.c"
#undef FLT
#undef X
#undef X2
#undef Y
#undef BLOCKRANK
#undef BLOCKSIZE
