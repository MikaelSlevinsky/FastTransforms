#include "fasttransforms.h"
#include "ftinternal.h"
#include "fmm.h"

size_t get_number_of_blocks(const size_t level) { return pow(2, level + 1) - 1; }

size_t get_h(const size_t level, const size_t L) { return pow(2, L - level - 1); }

size_t get_number_of_submatrices(const size_t level) { return 3 * get_number_of_blocks(level); }

size_t get_total_number_of_blocks(const size_t L) { return pow(2, L + 1) - (L + 2); }

size_t get_total_number_of_submatrices(const size_t L) { return 3 * get_total_number_of_blocks(L); }

void get_ij(size_t *ij, const size_t level, const size_t block, const size_t s, const size_t L)
{
  size_t h = get_h(level, L);
  ij[0] = 2 * block * s * h;
  ij[1] = ij[0] + 2 * s * h;
}

/*#if defined(FT_QUADMATH)
    #define FLT quadruple
    #define X(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, q)
    #include "fmm_source.c"
    #undef FLT
    #undef X
    #undef Y

    #define FLT long double
    #define X(name) FT_CONCAT(ft_, name, l)
    #define X2(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, l)
    #include "fmm_source.c"
    #undef FLT
    #undef X
    #undef X2
    #undef Y
#else
    #define FLT long double
    #define X(name) FT_CONCAT(ft_, name, l)
    #define X2(name) FT_CONCAT(ft_, name, l)
    #define Y(name) FT_CONCAT(, name, l)
    #include "fmm_source.c"
    #undef FLT
    #undef X
    #undef X2
    #undef Y
#endif*/


#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define FT_USE_DOUBLE
#include "fmm_source.c"
#undef FLT
#undef X
#undef FT_USE_DOUBLE

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define FT_USE_SINGLE
#include "fmm_source.c"
#undef FLT
#undef X
#undef FT_USE_SINGLE
