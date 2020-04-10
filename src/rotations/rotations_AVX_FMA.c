#include "rotations.h"

#ifdef __AVX__
    #ifdef __FMA__
        static inline void apply_givens_AVX_FMA(const double S, const double C, double * X, double * Y) {
            double4 x = vloadu4(X);
            double4 y = vloadu4(Y);

            vstoreu4(X, C*x + S*y);
            vstoreu4(Y, C*y - S*x);
        }

        static inline void apply_givens_t_AVX_FMA(const double S, const double C, double * X, double * Y) {
            double4 x = vloadu4(X);
            double4 y = vloadu4(Y);

            vstoreu4(X, C*x - S*y);
            vstoreu4(Y, C*y + S*x);
        }

        void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_TRI_HI2LO(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_AVX_FMA)
        }
        void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_TRI_LO2HI(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_t_AVX_FMA)
        }
    #else
        void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_tri_hi2lo_AVX(RP, m1, m2, A, 4);
        }
        void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_tri_lo2hi_AVX(RP, m1, m2, A, 4);
        }
    #endif
#else
    void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_hi2lo_SSE2(RP, m1, m2, A, S);
        kernel_tri_hi2lo_SSE2(RP, m1, m2+2, A+2, S);
    }
    void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_lo2hi_SSE2(RP, m1, m2, A, S);
        kernel_tri_lo2hi_SSE2(RP, m1, m2+2, A+2, S);
    }
#endif
