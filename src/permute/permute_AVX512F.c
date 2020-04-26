#include "permute.h"

#ifdef __AVX512F__
    void swap_warp_AVX512F(double * A, double * B, const int N) {
        SWAP_WARP_KERNEL(double, double8, 8, 4, vloadu8, vstoreu8)
    }
    void swap_warp_AVX512Ff(float * A, float * B, const int N) {
        SWAP_WARP_KERNEL(float, float16, 8, 4, vloadu16f, vstoreu16f)
    }
#else
    void swap_warp_AVX512F(double * A, double * B, const int N) {swap_warp_AVX(A, B, N);}
    void swap_warp_AVX512Ff(float * A, float * B, const int N) {swap_warp_AVXf(A, B, N);}
#endif
