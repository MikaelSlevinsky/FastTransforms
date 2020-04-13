#include "rotations.h"

#ifdef __AVX512F__
    static inline void apply_givens_AVX512F(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);

        vstoreu8(X, C*x + S*y);
        vstoreu8(Y, C*y - S*x);
    }
    static inline void apply_givens_t_AVX512F(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);

        vstoreu8(X, C*x - S*y);
        vstoreu8(Y, C*y + S*x);
    }
    void kernel_sph_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_SPH_HI2LO(double, double8, 8, 3, vloadu8, vstoreu8, apply_givens_AVX512F)
    }
    void kernel_sph_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_SPH_LO2HI(double, double8, 8, 3, vloadu8, vstoreu8, apply_givens_t_AVX512F)
    }
    void kernel_tri_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_HI2LO(double, double8, 8, 3, vloadu8, vstoreu8, apply_givens_AVX512F)
    }
    void kernel_tri_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_LO2HI(double, double8, 8, 3, vloadu8, vstoreu8, apply_givens_t_AVX512F)
    }
#else
    void kernel_sph_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_sph_hi2lo_AVX_FMA(RP, m1, m2, A, S);
        kernel_sph_hi2lo_AVX_FMA(RP, m1, m2+4, A+4, S);
    }
    void kernel_sph_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_sph_lo2hi_AVX_FMA(RP, m1, m2, A, S);
        kernel_sph_lo2hi_AVX_FMA(RP, m1, m2+4, A+4, S);
    }
    void kernel_tri_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_hi2lo_AVX_FMA(RP, m1, m2, A, S);
        kernel_tri_hi2lo_AVX_FMA(RP, m1, m2+4, A+4, S);
    }
    void kernel_tri_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_lo2hi_AVX_FMA(RP, m1, m2, A, S);
        kernel_tri_lo2hi_AVX_FMA(RP, m1, m2+4, A+4, S);
    }
#endif
