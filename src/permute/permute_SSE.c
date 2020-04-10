#include "permute.h"

#ifdef __SSE__
    void swap_warp_SSEf(float * A, float * B, const int N) {
        SWAP_WARP_KERNEL(float, float4, 4, 4, vloadu4f, vstoreu4f)
    }
#else
    void swap_warp_SSEf(float * A, float * B, const int N) {swap_warp_defaultf(A, B, N);}
#endif
