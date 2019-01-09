#ifndef ADC_H
#define ADC_H

/*
For assignments of the form:
for (int i = i.start; i < i.stop; i++)
    x[i] = ;
*/
typedef struct {
    int start;
    int stop;
} unitrange;

#define CONCAT(prefix, name, suffix) prefix ## name ## suffix

#define hash(m,n) hash[(m)+(n)*M]
#define hierarchicalmatrices(m,n) hierarchicalmatrices[(m)+(n)*M]
#define densematrices(m,n) densematrices[(m)+(n)*M]
#define lowrankmatrices(m,n) lowrankmatrices[(m)+(n)*M]

#define FLT float
#define X(name) CONCAT(, name, f)
#include "arrow_source.h"
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "arrow_source.h"
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "arrow_source.h"
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#undef FLT
#undef X

#define FLT __float128
#define X(name) CONCAT(, name, q)
#include "arrow_source.h"
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#undef FLT
#undef X

#endif // ADC_H
