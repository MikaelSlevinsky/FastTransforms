#include "recurrence.h"

#ifdef __aarch64__
    void horner_NEON(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        HORNER_KERNEL(double, float64x2_t, 2, 16, vld1q_f64, vst1q_f64, vmuladd_f64, vall_f64)
    }
    void horner_NEONf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        HORNER_KERNEL(float, float32x4_t, 4, 16, vld1q_f32, vst1q_f32, vmuladd_f32, vall_f32)
    }
    void clenshaw_NEON(const int n, const double * c, const int incc, const int m, double * x, double * f) {
        CLENSHAW_KERNEL(double, float64x2_t, 2, 8, vld1q_f64, vst1q_f64, vmuladd_f64, vall_f64)
    }
    void clenshaw_NEONf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
        CLENSHAW_KERNEL(float, float32x4_t, 4, 8, vld1q_f32, vst1q_f32, vmuladd_f32, vall_f32)
    }
    void orthogonal_polynomial_clenshaw_NEON(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {
        ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(double, float64x2_t, 2, 8, vld1q_f64, vst1q_f64, vmuladd_f64, vmulsub_f64, vall_f64)
    }
    void orthogonal_polynomial_clenshaw_NEONf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {
        ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(float, float32x4_t, 2, 8, vld1q_f32, vst1q_f32, vmuladd_f32, vmulsub_f32, vall_f32)
    }
#else
    void horner_NEON(const int n, const double * c, const int incc, const int m, double * x, double * f) {horner_default(n, c, incc, m, x, f);}
    void horner_NEONf(const int n, const float * c, const int incc, const int m, float * x, float * f) {horner_defaultf(n, c, incc, m, x, f);}
    void clenshaw_NEON(const int n, const double * c, const int incc, const int m, double * x, double * f) {clenshaw_default(n, c, incc, m, x, f);}
    void clenshaw_NEONf(const int n, const float * c, const int incc, const int m, float * x, float * f) {clenshaw_defaultf(n, c, incc, m, x, f);}
    void orthogonal_polynomial_clenshaw_NEON(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {orthogonal_polynomial_clenshaw_default(n, c, incc, A, B, C, m, x, phi0, f);}
    void orthogonal_polynomial_clenshaw_NEONf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {orthogonal_polynomial_clenshaw_defaultf(n, c, incc, A, B, C, m, x, phi0, f);}
#endif
