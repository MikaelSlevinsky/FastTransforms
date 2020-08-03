#include "rotations.h"

#ifdef __aarch64__
    static inline void apply_givens_NEON(const double S, const double C, double * X, double * Y) {
        float64x2_t x = vld1q_f64(X);
        float64x2_t y = vld1q_f64(Y);
        vst1q_f64(X, vall_f64(C)*x + vall_f64(S)*y);
        vst1q_f64(Y, vall_f64(C)*y - vall_f64(S)*x);
    }
    static inline void apply_givens_t_NEON(const double S, const double C, double * X, double * Y) {
        float64x2_t x = vld1q_f64(X);
        float64x2_t y = vld1q_f64(Y);
        vst1q_f64(X, vall_f64(C)*x - vall_f64(S)*y);
        vst1q_f64(Y, vall_f64(C)*y + vall_f64(S)*x);
    }
    static inline void apply_givens_NEONc(const double S, const double C, ft_complex * X, ft_complex * Y) {
        apply_givens_NEON(S, C, (double *) X, (double *) Y);
    }
    static inline void apply_givens_t_NEONc(const double S, const double C, ft_complex * X, ft_complex * Y) {
        apply_givens_t_NEON(S, C, (double *) X, (double *) Y);
    }
    void kernel_sph_hi2lo_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_SPH_HI2LO(double, double2, 2, 3, vld1q_f64, vst1q_f64, vmuladd, vmulsub, vall_f64, apply_givens_NEON)
    }
    void kernel_sph_lo2hi_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_SPH_LO2HI(double, double2, 2, 3, vld1q_f64, vst1q_f64, vmuladd, vmulsub, vall_f64, apply_givens_t_NEON)
    }
    void kernel_tri_hi2lo_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_HI2LO(double, double2, 2, 3, vld1q_f64, vst1q_f64, vmuladd, vmulsub, vall_f64, apply_givens_NEON)
    }
    void kernel_tri_lo2hi_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_TRI_LO2HI(double, double2, 2, 3, vld1q_f64, vst1q_f64, vmuladd, vmulsub, vall_f64, apply_givens_t_NEON)
    }
    void kernel_disk_hi2lo_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_DISK_HI2LO(double, double2, 2, 3, vld1q_f64, vst1q_f64, vmuladd, vmulsub, vall_f64, apply_givens_NEON)
    }
    void kernel_disk_lo2hi_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        KERNEL_DISK_LO2HI(double, double2, 2, 3, vld1q_f64, vst1q_f64, vmuladd, vmulsub, vall_f64, apply_givens_t_NEON)
    }
    void kernel_spinsph_hi2lo_NEON(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
        int n = SRP->n, s = SRP->s;
        int as = abs(s), am = abs(m);
        if (m*s >= 0)
            for (int j = MIN(am, as)-1; j >= 0; j--)
                for (int k = n-2-abs(am-as)-j; k >= 0; k--)
                    apply_givens_NEONc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
        else
            for (int j = MIN(am, as)-1; j >= 0; j--)
                for (int k = n-2-abs(am-as)-j; k >= 0; k--)
                    apply_givens_t_NEONc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
        for (int j = abs(am-as)-2; j >= (am+as)%2; j -= 2)
            for (int k = n-3-j; k >= 0; k--)
                apply_givens_NEONc(SRP->s1(k, j), SRP->c1(k, j), A+k*S, A+(k+2)*S);
    }
    void kernel_spinsph_lo2hi_NEON(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
        int n = SRP->n, s = SRP->s;
        int as = abs(s), am = abs(m);
        for (int j = (am+as)%2; j <= abs(am-as)-2; j += 2)
            for (int k = 0; k <= n-3-j; k++)
                apply_givens_t_NEONc(SRP->s1(k, j), SRP->c1(k, j), A+k*S, A+(k+2)*S);
        if (m*s >= 0)
            for (int j = 0; j < MIN(am, as); j++)
                for (int k = 0; k <= n-2-abs(am-as)-j; k++)
                    apply_givens_t_NEONc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
        else
            for (int j = 0; j < MIN(am, as); j++)
                for (int k = 0; k <= n-2-abs(am-as)-j; k++)
                    apply_givens_NEONc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
    }
#else
    void kernel_sph_hi2lo_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_sph_hi2lo_default(RP, m1, m2, A, S);
        kernel_sph_hi2lo_default(RP, m1, m2, A+1, S);
    }
    void kernel_sph_lo2hi_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_sph_lo2hi_default(RP, m1, m2, A, S);
        kernel_sph_lo2hi_default(RP, m1, m2, A+1, S);
    }
    void kernel_tri_hi2lo_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_hi2lo_default(RP, m1, m2, A, S);
        kernel_tri_hi2lo_default(RP, m1, m2+1, A+1, S);
    }
    void kernel_tri_lo2hi_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_lo2hi_default(RP, m1, m2, A, S);
        kernel_tri_lo2hi_default(RP, m1, m2+1, A+1, S);
    }
    void kernel_disk_hi2lo_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_disk_hi2lo_default(RP, m1, m2, A, S);
        kernel_disk_hi2lo_default(RP, m1, m2, A+1, S);
    }
    void kernel_disk_lo2hi_NEON(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_disk_lo2hi_default(RP, m1, m2, A, S);
        kernel_disk_lo2hi_default(RP, m1, m2, A+1, S);
    }
    void kernel_spinsph_hi2lo_NEON(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
        kernel_spinsph_hi2lo_default(SRP, m, A, S);
    }
    void kernel_spinsph_lo2hi_NEON(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
        kernel_spinsph_lo2hi_default(SRP, m, A, S);
    }
#endif
