#include "fasttransforms.h"
#include "ftinternal.h"

#define FLT float
#define X(name) CONCAT(, name, f)
#define BLOCKRANK 2*((int) floor(-log(X(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "arrow_source.c"
#include "tridiagonal_source.c"
#include "hierarchical_source.c"
#undef FLT
#undef X
#undef BLOCKRANK
#undef BLOCKSIZE

#define FLT double
#define X(name) CONCAT(, name, )
#define BLOCKRANK 2*((int) floor(-log(X(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "arrow_source.c"
#include "tridiagonal_source.c"
#include "hierarchical_source.c"
#undef FLT
#undef X
#undef BLOCKRANK
#undef BLOCKSIZE

#define FLT long double
#define X(name) CONCAT(, name, l)
#define BLOCKRANK 2*((int) floor(-log(X(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "arrow_source.c"
#include "tridiagonal_source.c"
#include "hierarchical_source.c"
#undef FLT
#undef X
#undef BLOCKRANK
#undef BLOCKSIZE

#define FLT quadruple
#define X(name) CONCAT(, name, q)
#define BLOCKRANK 2*((int) floor(-log(X(eps)())/3.525494348078172))
#define BLOCKSIZE 4*BLOCKRANK
#include "arrow_source.c"
#include "tridiagonal_source.c"
#include "hierarchical_source.c"
#undef FLT
#undef X
#undef BLOCKRANK
#undef BLOCKSIZE
