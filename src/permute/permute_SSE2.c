#include "permute.h"

#ifdef __SSE2__
    void swap_warp_SSE2(double * A, double * B, const int N) {
        SWAP_WARP_KERNEL(double, double2, 2, 4, vloadu2, vstoreu2)
    }
#else
    void swap_warp_SSE2(double * A, double * B, const int N) {swap_warp_default(A, B, N);}
#endif
