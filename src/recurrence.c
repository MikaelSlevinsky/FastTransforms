#include "fasttransforms.h"
#include "ftinternal.h"

void ft_horner(const int n, const double * c, const int incc, const int m, double * x, double * f) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return horner_AVX512F(n, c, incc, m, x, f);
    else if (simd.avx) {
        if (simd.fma)
            return horner_AVX_FMA(n, c, incc, m, x, f);
        else
            return horner_AVX(n, c, incc, m, x, f);
    }
    else if (simd.sse2)
        return horner_SSE2(n, c, incc, m, x, f);
    else if (simd.neon)
        return horner_NEON(n, c, incc, m, x, f);
    else
        return horner_default(n, c, incc, m, x, f);
}

void ft_hornerf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return horner_AVX512Ff(n, c, incc, m, x, f);
    else if (simd.avx) {
        if (simd.fma)
            return horner_AVX_FMAf(n, c, incc, m, x, f);
        else
            return horner_AVXf(n, c, incc, m, x, f);
    }
    else if (simd.sse)
        return horner_SSEf(n, c, incc, m, x, f);
    else if (simd.neon)
        return horner_NEONf(n, c, incc, m, x, f);
    else
        return horner_defaultf(n, c, incc, m, x, f);
}

void ft_clenshaw(const int n, const double * c, const int incc, const int m, double * x, double * f) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return clenshaw_AVX512F(n, c, incc, m, x, f);
    else if (simd.avx) {
        if (simd.fma)
            return clenshaw_AVX_FMA(n, c, incc, m, x, f);
        else
            return clenshaw_AVX(n, c, incc, m, x, f);
    }
    else if (simd.sse2)
        return clenshaw_SSE2(n, c, incc, m, x, f);
    else if (simd.neon)
        return clenshaw_NEON(n, c, incc, m, x, f);
    else
        return clenshaw_default(n, c, incc, m, x, f);
}

void ft_clenshawf(const int n, const float * c, const int incc, const int m, float * x, float * f) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return clenshaw_AVX512Ff(n, c, incc, m, x, f);
    else if (simd.avx) {
        if (simd.fma)
            return clenshaw_AVX_FMAf(n, c, incc, m, x, f);
        else
            return clenshaw_AVXf(n, c, incc, m, x, f);
    }
    else if (simd.sse)
        return clenshaw_SSEf(n, c, incc, m, x, f);
    else if (simd.neon)
        return clenshaw_NEONf(n, c, incc, m, x, f);
    else
        return clenshaw_defaultf(n, c, incc, m, x, f);
}

void ft_orthogonal_polynomial_clenshaw(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return orthogonal_polynomial_clenshaw_AVX512F(n, c, incc, A, B, C, m, x, phi0, f);
    else if (simd.avx) {
        if (simd.fma)
            return orthogonal_polynomial_clenshaw_AVX_FMA(n, c, incc, A, B, C, m, x, phi0, f);
        else
            return orthogonal_polynomial_clenshaw_AVX(n, c, incc, A, B, C, m, x, phi0, f);
    }
    else if (simd.sse2)
        return orthogonal_polynomial_clenshaw_SSE2(n, c, incc, A, B, C, m, x, phi0, f);
    else if (simd.neon)
        return orthogonal_polynomial_clenshaw_NEON(n, c, incc, A, B, C, m, x, phi0, f);
    else
        return orthogonal_polynomial_clenshaw_default(n, c, incc, A, B, C, m, x, phi0, f);
}

void ft_orthogonal_polynomial_clenshawf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f) {
    ft_simd simd = get_simd();
    if (simd.avx512f)
        return orthogonal_polynomial_clenshaw_AVX512Ff(n, c, incc, A, B, C, m, x, phi0, f);
    else if (simd.avx) {
        if (simd.fma)
            return orthogonal_polynomial_clenshaw_AVX_FMAf(n, c, incc, A, B, C, m, x, phi0, f);
        else
            return orthogonal_polynomial_clenshaw_AVXf(n, c, incc, A, B, C, m, x, phi0, f);
    }
    else if (simd.sse)
        return orthogonal_polynomial_clenshaw_SSEf(n, c, incc, A, B, C, m, x, phi0, f);
    else if (simd.neon)
        return orthogonal_polynomial_clenshaw_NEONf(n, c, incc, A, B, C, m, x, phi0, f);
    else
        return orthogonal_polynomial_clenshaw_defaultf(n, c, incc, A, B, C, m, x, phi0, f);
}
