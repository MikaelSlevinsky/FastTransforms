// Staging ground for determining the minimal assembly Givens instructions.

#include <immintrin.h>

#if __AVX512F__
    #define VECTOR_SIZE_8 8
    #define ALIGN_SIZE VECTOR_SIZE_8
    typedef double double8 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
    #define vall8(x) ((double8) _mm512_set1_pd(x))
    #define vload8(v) ((double8) _mm512_load_pd(v))
    #define vstore8(u, v) (_mm512_store_pd(u, v))
    #define vsqrt8(x) ((double8) _mm512_sqrt_pd(x))
#endif
#if __AVX__
    #define VECTOR_SIZE_4 4
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_4
    #endif
    typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
    #define vall4(x) ((double4) _mm256_set1_pd(x))
    #define vload4(v) ((double4) _mm256_load_pd(v))
    #define vstore4(u, v) (_mm256_store_pd(u, v))
    #define vsqrt4(x) ((double4) _mm256_sqrt_pd(x))
#endif
#if __SSE2__
    #define VECTOR_SIZE_2 2
    #ifndef ALIGN_SIZE
        #define ALIGN_SIZE VECTOR_SIZE_2
    #endif
    typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
    #define vall2(x) ((double2) _mm_set1_pd(x))
    #define vload2(v) ((double2) _mm_load_pd(v))
    #define vstore2(u, v) (_mm_store_pd(u, v))
    #define vsqrt2(x) ((double2) _mm_sqrt_pd(x))
#endif

void swap(double * A, double * B, const int N) {
    double tmp;
    for (int i = 0; i < N; i++) {
        tmp = A[i];
        A[i] = B[i];
        B[i] = tmp;
    }
}

void apply_givens(const double S, const double C, double * X, double * Y) {
    double x = C*X[0] + S*Y[0];
    double y = C*Y[0] - S*X[0];

    X[0] = x;
    Y[0] = y;
}

#if __SSE2__
    void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, C*x + S*y);
        vstore2(Y, C*y - S*x);
    }
#else
    void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        apply_givens(S, C, X, Y);
        apply_givens(S, C, X+1, Y+1);
    }
#endif


#if __AVX__
    void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, C*x + S*y);
        vstore4(Y, C*y - S*x);
    }
#else
    void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        apply_givens_SSE(S, C, X, Y);
        apply_givens_SSE(S, C, X+2, Y+2);
    }
#endif

#if __AVX512F__
    void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, C*x + S*y);
        vstore8(Y, C*y - S*x);
    }
#else
    void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        apply_givens_AVX(S, C, X, Y);
        apply_givens_AVX(S, C, X+4, Y+4);
    }
#endif


void ft_kernel_sph_hi2lo_AVX512(const int n, const int m, double * A) {
    double2 scnum2;
    double4 scnum4, scden4;
    double sc[4];
    for (int l = n-3-m; l >= 0; l--) {
        scnum2 = (double2) {(l+1)*(l+2), (2*m+2)*(2*l+2*m+5)};
        vstore2(sc, vsqrt2(scnum2/((l+2*m+3)*(l+2*m+4))));
        apply_givens_SSE(sc[0], sc[1], A+8*l+2, A+8*l+18);
    }
    for (int l = n-7-m; l >= 0; l--) {
        scnum2 = (double2) {(l+1)*(l+2), (2*m+10)*(2*l+2*m+13)};
        vstore2(sc, vsqrt2(scnum2/((l+2*m+11)*(l+2*m+12))));
        apply_givens_SSE(sc[0], sc[1], A+8*l+6, A+8*l+22);
    }
    for (int j = m+2; j >= m; j -= 2)
        for (int l = n-3-j; l >= 0; l--) {
            scnum2 = (double2) {(l+1)*(l+2), (2*j+2)*(2*l+2*j+5)};
            vstore2(sc, vsqrt2(scnum2/((l+2*j+3)*(l+2*j+4))));
            apply_givens_AVX(sc[0], sc[1], A+8*l+4, A+8*l+20);
        }
    for (int j = m-2; j >= 0; j -= 2) {
        for (int l = n-3-j; l >= 0; l -= 2) {
            scnum4 = (double4) {(l+1)*(l+2), (2*j+2)*(2*l+2*j+5), l*(l+1), (2*j+2)*(2*l+2*j+3)};
            scden4 = (double4) {(l+2*j+3)*(l+2*j+4), (l+2*j+3)*(l+2*j+4), (l+2*j+2)*(l+2*j+3), (l+2*j+2)*(l+2*j+3)};
            vstore4(sc, vsqrt4(scnum4/scden4));
            apply_givens_AVX512(sc[0], sc[1], A+8*l, A+8*l+16);
            apply_givens_AVX512(sc[2], sc[3], A+8*l-8, A+8*l+8);
        }
        if (n-3-j % 2 == 0) {
            scnum2 = (double2) {2, (2*j+2)*(2*j+5)};
            vstore2(sc, vsqrt2(scnum2/((2*j+3)*(2*j+4))));
            apply_givens_AVX512(sc[0], sc[1], A, A+16);
        }
    }
}
