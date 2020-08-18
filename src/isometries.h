#ifndef ISOMETRIES_H
#define ISOMETRIES_H

#include "fasttransforms.h"
#include "ftinternal.h"

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "isometries_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "isometries_source.c"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "isometries_source.c"
#undef FLT
#undef X
#undef Y

#if defined(FT_QUADMATH)
	#define FLT quadruple
	#define X(name) FT_CONCAT(ft_, name, q)
	#define Y(name) FT_CONCAT(, name, q)
	#include "isometries_source.c"
	#undef FLT
	#undef X
	#undef Y
#endif

#endif
