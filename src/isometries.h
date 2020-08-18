#ifndef ISOMETRIES_H
#define ISOMETRIES_H

#include "fasttransforms.h"
#include "ftinternal.h"

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "isometries_source.c"
#undef FLT
#undef X
#undef Y

#endif
