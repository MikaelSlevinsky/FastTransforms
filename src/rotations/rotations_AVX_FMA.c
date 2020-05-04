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
        static inline void apply_givens_AVX_FMAc(const double S, const double C, ft_complex * X, ft_complex * Y) {
            apply_givens_AVX_FMA(S, C, (double *) X, (double *) Y);
        }
        static inline void apply_givens_t_AVX_FMAc(const double S, const double C, ft_complex * X, ft_complex * Y) {
            apply_givens_t_AVX_FMA(S, C, (double *) X, (double *) Y);
        }
        static inline void apply_twisted_givens_AVX_FMAc(const double S, const double C, ft_complex * X, ft_complex * Y) {
            double * XD = (double *) X;
            double * YD = (double *) Y;
            double4 x = vloadu4(XD);
            double4 y = vloadu4(YD);
            vstoreu4(XD, vfmas4(vall4(C), x,  S*y));
            vstoreu4(YD, vfmas4(vall4(C), y, -S*x));
        }
        static inline void apply_twisted_givens_t_AVX_FMAc(const double S, const double C, ft_complex * X, ft_complex * Y) {
            double * XD = (double *) X;
            double * YD = (double *) Y;
            double4 x = vloadu4(XD);
            double4 y = vloadu4(YD);
            vstoreu4(XD, vfmas4(vall4(C), x, -S*y));
            vstoreu4(YD, vfmas4(vall4(C), y,  S*x));
        }
        void kernel_sph_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_SPH_HI2LO(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_AVX_FMA)
        }
        void kernel_sph_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_SPH_LO2HI(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_t_AVX_FMA)
        }
        void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_TRI_HI2LO(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_AVX_FMA)
        }
        void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_TRI_LO2HI(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_t_AVX_FMA)
        }
        void kernel_disk_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_DISK_HI2LO(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_AVX_FMA)
        }
        void kernel_disk_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            KERNEL_DISK_LO2HI(double, double4, 4, 3, vloadu4, vstoreu4, apply_givens_t_AVX_FMA)
        }
        void kernel_spinsph_hi2lo_AVX_FMA(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
            int n = SRP->n, s = SRP->s;
            int as = abs(s), am = abs(m);
            if (s >= 0)
                for (int j = MIN(am, as)-1; j >= 0; j--)
                    for (int k = n-2-abs(am-as)-j; k >= 0; k--)
                        apply_twisted_givens_AVX_FMAc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
            else
                for (int j = MIN(am, as)-1; j >= 0; j--)
                    for (int k = n-2-abs(am-as)-j; k >= 0; k--)
                        apply_twisted_givens_t_AVX_FMAc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
            for (int j = abs(am-as)-2; j >= (am+as)%2; j -= 2)
                for (int k = n-3-j; k >= 0; k--)
                    apply_givens_AVX_FMAc(SRP->s1(k, j), SRP->c1(k, j), A+k*S, A+(k+2)*S);
        }
        void kernel_spinsph_lo2hi_AVX_FMA(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
            int n = SRP->n, s = SRP->s;
            int as = abs(s), am = abs(m);
            for (int j = (am+as)%2; j <= abs(am-as)-2; j += 2)
                for (int k = 0; k <= n-3-j; k++)
                    apply_givens_t_AVX_FMAc(SRP->s1(k, j), SRP->c1(k, j), A+k*S, A+(k+2)*S);
            if (s >= 0)
                for (int j = 0; j < MIN(am, as); j++)
                    for (int k = 0; k <= n-2-abs(am-as)-j; k++)
                        apply_twisted_givens_t_AVX_FMAc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
            else
                for (int j = 0; j < MIN(am, as); j++)
                    for (int k = 0; k <= n-2-abs(am-as)-j; k++)
                        apply_twisted_givens_AVX_FMAc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
        }
    #else
        void kernel_sph_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_sph_hi2lo_AVX(RP, m1, m2, A, S);
        }
        void kernel_sph_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_sph_lo2hi_AVX(RP, m1, m2, A, S);
        }
        void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_tri_hi2lo_AVX(RP, m1, m2, A, S);
        }
        void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_tri_lo2hi_AVX(RP, m1, m2, A, S);
        }
        void kernel_disk_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_disk_hi2lo_AVX(RP, m1, m2, A, S);
        }
        void kernel_disk_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
            kernel_disk_lo2hi_AVX(RP, m1, m2, A, S);
        }
        void kernel_spinsph_hi2lo_AVX_FMA(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
            kernel_spinsph_hi2lo_AVX(SRP, m, A, S);
        }
        void kernel_spinsph_lo2hi_AVX_FMA(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
            kernel_spinsph_lo2hi_AVX(SRP, m, A, S);
        }
    #endif
#else
    void kernel_sph_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_sph_hi2lo_SSE2(RP, m1, m2, A, S);
        kernel_sph_hi2lo_SSE2(RP, m1, m2+2, A+2, S);
    }
    void kernel_sph_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_sph_lo2hi_SSE2(RP, m1, m2, A, S);
        kernel_sph_lo2hi_SSE2(RP, m1, m2+2, A+2, S);
    }
    void kernel_tri_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_hi2lo_SSE2(RP, m1, m2, A, S);
        kernel_tri_hi2lo_SSE2(RP, m1, m2+2, A+2, S);
    }
    void kernel_tri_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_tri_lo2hi_SSE2(RP, m1, m2, A, S);
        kernel_tri_lo2hi_SSE2(RP, m1, m2+2, A+2, S);
    }
    void kernel_disk_hi2lo_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_disk_hi2lo_SSE2(RP, m1, m2, A, S);
        kernel_disk_hi2lo_SSE2(RP, m1, m2+2, A+2, S);
    }
    void kernel_disk_lo2hi_AVX_FMA(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
        kernel_disk_lo2hi_SSE2(RP, m1, m2, A, S);
        kernel_disk_lo2hi_SSE2(RP, m1, m2+2, A+2, S);
    }
#endif
