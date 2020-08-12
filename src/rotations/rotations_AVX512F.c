#include "rotations.h"

#ifdef __AVX512F__
    static inline void apply_givens_AVX512F(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);
        vstoreu8(X, vfma8(vall8(C), x, S*y));
        vstoreu8(Y, vfms8(vall8(C), y, S*x));
    }
    static inline void apply_givens_t_AVX512F(const double S, const double C, double * X, double * Y) {
        double8 x = vloadu8(X);
        double8 y = vloadu8(Y);
        vstoreu8(X, vfms8(vall8(C), x, S*y));
        vstoreu8(Y, vfma8(vall8(C), y, S*x));
    }
    void kernel_sph_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_SPH_HI2LO(double, double8, 8, 3, vloadu8, vstoreu8, vfma8, vfms8, vall8, apply_givens_AVX512F)
    }
    void kernel_sph_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_SPH_LO2HI(double, double8, 8, 3, vloadu8, vstoreu8, vfma8, vfms8, vall8, apply_givens_t_AVX512F)
    }
    void kernel_tri_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_HI2LO(double, double8, 8, 3, vloadu8, vstoreu8, vfma8, vfms8, vall8, apply_givens_AVX512F)
    }
    void kernel_tri_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_LO2HI(double, double8, 8, 3, vloadu8, vstoreu8, vfma8, vfms8, vall8, apply_givens_t_AVX512F)
    }
    void kernel_disk_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_DISK_HI2LO(double, double8, 8, 3, vloadu8, vstoreu8, vfma8, vfms8, vall8, apply_givens_AVX512F)
    }
    void kernel_disk_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_DISK_LO2HI(double, double8, 8, 3, vloadu8, vstoreu8, vfma8, vfms8, vall8, apply_givens_t_AVX512F)
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
    void kernel_disk_hi2lo_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_disk_hi2lo_AVX_FMA(RP, m1, m2, A, S);
        kernel_disk_hi2lo_AVX_FMA(RP, m1, m2+4, A+4, S);
    }
    void kernel_disk_lo2hi_AVX512F(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_disk_lo2hi_AVX_FMA(RP, m1, m2, A, S);
        kernel_disk_lo2hi_AVX_FMA(RP, m1, m2+4, A+4, S);
    }
#endif
