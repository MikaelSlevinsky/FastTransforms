#include "tridiagonal.h"
#include "ftinternal.h"

#define FLT float
#define X(name) CONCAT(, name, f)
#include "tridiagonal_source.c"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "tridiagonal_source.c"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "tridiagonal_source.c"
#undef FLT
#undef X

#define FLT quadruple
#define X(name) CONCAT(, name, q)
#include "tridiagonal_source.c"
#undef FLT
#undef X
