#ifndef TRIDIAGONAL_H
#define TRIDIAGONAL_H

#define CONCAT(prefix, name, suffix) prefix ## name ## suffix

#define FLT float
#define X(name) CONCAT(, name, f)
#include "tridiagonal_source.h"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "tridiagonal_source.h"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "tridiagonal_source.h"
#undef FLT
#undef X

#define FLT __float128
#define X(name) CONCAT(, name, q)
#include "tridiagonal_source.h"
#undef FLT
#undef X

#endif // TRIDIAGONAL_H
