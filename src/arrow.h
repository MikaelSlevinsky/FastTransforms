#ifndef ARROW_H
#define ARROW_H

#define CONCAT(prefix, name, suffix) prefix ## name ## suffix

#define FLT float
#define X(name) CONCAT(, name, f)
#include "arrow_source.h"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "arrow_source.h"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "arrow_source.h"
#undef FLT
#undef X

#define FLT __float128
#define X(name) CONCAT(, name, q)
#include "arrow_source.h"
#undef FLT
#undef X

#endif // ARROW_H
