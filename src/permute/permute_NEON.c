#include "permute.h"

#ifdef __aarch64__
    void swap_warp_NEON(double * A, double * B, const int N) {
        SWAP_WARP_KERNEL(double, float64x2_t, 2, 4, vld1q_f64, vst1q_f64)
    }
    void swap_warp_NEONf(float * A, float * B, const int N) {
        SWAP_WARP_KERNEL(float, float32x4_t, 4, 4, vld1q_f32, vst1q_f32)
    }
#else
    void swap_warp_NEON(double * A, double * B, const int N) {swap_warp_default(A, B, N);}
    void swap_warp_NEONf(float * A, float * B, const int N) {swap_warp_defaultf(A, B, N);}
#endif
