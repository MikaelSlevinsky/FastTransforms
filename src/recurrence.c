#include "fasttransforms.h"
#include "ftinternal.h"

static inline ft_simd get_simd() {
    unsigned int eax, ebx, ecx, edx;
    unsigned int eax1, ebx1, ecx1, edx1;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    __get_cpuid_count(7, 0, &eax1, &ebx1, &ecx1, &edx1);
    return (ft_simd) {edx & bit_SSE, edx & bit_SSE2, ecx & bit_AVX, ecx & bit_FMA, ebx1 & bit_AVX512F};
}

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
    else
        return clenshaw_defaultf(n, c, incc, m, x, f);
}
