#include "fasttransforms.h"
#include "ftinternal.h"

#define TB_EIGEN_BLOCKSIZE 128
#define TDC_EIGEN_BLOCKSIZE 128

#if defined(FT_QUADMATH)
    #define FLT quadruple
    #define X(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, q)
    #define BLOCKRANK 2*((int) floor(-log(Y(eps)())/2.271667761226165))
    #define BLOCKSIZE 4*BLOCKRANK
    #include "tridiagonal_source.c"
    #include "hierarchical_source.c"
    #include "banded_source.c"
    #include "dprk_source.c"
    #include "tdc_source.c"
    #undef FLT
    #undef X
    #undef Y
    #undef BLOCKRANK
    #undef BLOCKSIZE

    #define FLT long double
    #define X(name) FT_CONCAT(ft_, name, l)
    #define X2(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, l)
    #define BLOCKRANK 2*((int) floor(-log(Y(eps)())/2.271667761226165))
    #define BLOCKSIZE 4*BLOCKRANK
    #include "tridiagonal_source.c"
    #include "hierarchical_source.c"
    #include "banded_source.c"
    #include "dprk_source.c"
    #include "tdc_source.c"
    #include "drop_precision.c"
    #undef FLT
    #undef X
    #undef X2
    #undef Y
    #undef BLOCKRANK
    #undef BLOCKSIZE
#else
    #define FLT long double
    #define X(name) FT_CONCAT(ft_, name, l)
    #define X2(name) FT_CONCAT(ft_, name, l)
    #define Y(name) FT_CONCAT(, name, l)
    #define BLOCKRANK 2*((int) floor(-log(Y(eps)())/2.271667761226165))
    #define BLOCKSIZE 4*BLOCKRANK
    #include "tridiagonal_source.c"
    #include "hierarchical_source.c"
    #include "banded_source.c"
    #include "dprk_source.c"
    #include "tdc_source.c"
    #include "drop_precision.c"
    #undef FLT
    #undef X
    #undef X2
    #undef Y
    #undef BLOCKRANK
    #undef BLOCKSIZE
#endif

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define X2(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, )
#define BLOCKRANK 2*((int) floor(-log(Y(eps)())/2.271667761226165))
#define BLOCKSIZE 4*BLOCKRANK
#define FT_USE_CBLAS_D
#include "tridiagonal_source.c"
#include "hierarchical_source.c"
#include "banded_source.c"
#include "dprk_source.c"
#include "tdc_source.c"
#include "drop_precision.c"
#undef FLT
#undef X
#undef X2
#undef Y
#undef BLOCKRANK
#undef BLOCKSIZE
#undef FT_USE_CBLAS_D

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define X2(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, f)
#define BLOCKRANK 2*((int) floor(-log(Y(eps)())/2.271667761226165))
#define BLOCKSIZE 4*BLOCKRANK
#define FT_USE_CBLAS_S
#include "tridiagonal_source.c"
#include "hierarchical_source.c"
#include "banded_source.c"
#include "dprk_source.c"
#include "tdc_source.c"
#include "drop_precision.c"
#undef FLT
#undef X
#undef X2
#undef Y
#undef BLOCKRANK
#undef BLOCKSIZE
#undef FT_USE_CBLAS_S

#undef TB_EIGEN_BLOCKSIZE
#undef TDC_EIGEN_BLOCKSIZE
