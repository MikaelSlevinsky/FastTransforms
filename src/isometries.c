#include "fasttransforms.h"
#include "ftinternal.h"

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "isometries_source.c"
#undef FLT
#undef X
#undef Y
