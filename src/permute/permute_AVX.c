#include "permute.h"

#ifdef __AVX__
    void swap_warp_AVX(double * A, double * B, const int N) {
        SWAP_WARP_KERNEL(double, double4, 4, 4, vloadu4, vstoreu4)
    }
    void swap_warp_AVXf(float * A, float * B, const int N) {
        SWAP_WARP_KERNEL(float, float8, 8, 4, vloadu8f, vstoreu8f)
    }
#else
    void swap_warp_AVX(double * A, double * B, const int N) {swap_warp_SSE2(A, B, N);}
    void swap_warp_AVXf(float * A, float * B, const int N) {swap_warp_SSEf(A, B, N);}
#endif
