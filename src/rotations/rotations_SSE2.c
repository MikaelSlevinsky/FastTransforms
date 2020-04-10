#include "rotations.h"

#ifdef __SSE2__
    static inline void apply_givens_SSE2(const double S, const double C, double * X, double * Y) {
        double2 x = vloadu2(X);
        double2 y = vloadu2(Y);

        vstoreu2(X, C*x + S*y);
        vstoreu2(Y, C*y - S*x);
    }

    static inline void apply_givens_t_SSE2(const double S, const double C, double * X, double * Y) {
        double2 x = vloadu2(X);
        double2 y = vloadu2(Y);

        vstoreu2(X, C*x - S*y);
        vstoreu2(Y, C*y + S*x);
    }

    void kernel_tri_hi2lo_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_HI2LO(double, double2, 2, 3, vloadu2, vstoreu2, apply_givens_SSE2)
    }
    void kernel_tri_lo2hi_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_LO2HI(double, double2, 2, 3, vloadu2, vstoreu2, apply_givens_t_SSE2)
    }
#else
    void kernel_tri_hi2lo_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_hi2lo_default(RP, m1, m2, A, S);
        kernel_tri_hi2lo_default(RP, m1, m2+1, A+1, S);
    }
    void kernel_tri_lo2hi_SSE2(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_lo2hi_default(RP, m1, m2, A, S);
        kernel_tri_lo2hi_default(RP, m1, m2+1, A+1, S);
    }
#endif
