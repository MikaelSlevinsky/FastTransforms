#ifndef TDC_H
#define TDC_H

/*
For assignments of the form:
for (int i = i.start; i < i.stop; i++)
    x[i] = ;
*/
typedef struct {
    int start;
    int stop;
} unitrange;

#define hash(m,n) hash[(m)+(n)*M]
#define hierarchicalmatrices(m,n) hierarchicalmatrices[(m)+(n)*M]
#define densematrices(m,n) densematrices[(m)+(n)*M]
#define lowrankmatrices(m,n) lowrankmatrices[(m)+(n)*M]

#define FLT __float128
#define X(name) FT_CONCAT(ft_, name, q)
#define Y(name) FT_CONCAT(, name, q)
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#include "banded_source.h"
#include "dprk_source.h"
#include "tdc_source.h"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#include "banded_source.h"
#include "dprk_source.h"
#include "tdc_source.h"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define X2(name) FT_CONCAT(ft_, name, q)
#define Y(name) FT_CONCAT(, name, )
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#include "banded_source.h"
#include "dprk_source.h"
#include "tdc_source.h"
#include "tdc_source2.h"
#undef FLT
#undef X
#undef X2
#undef Y

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define X2(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, f)
#include "tridiagonal_source.h"
#include "hierarchical_source.h"
#include "banded_source.h"
#include "dprk_source.h"
#include "tdc_source.h"
#include "tdc_source2.h"
#undef FLT
#undef X
#undef X2
#undef Y

#endif // TDC_H
