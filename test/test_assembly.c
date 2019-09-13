// Staging ground for determining the minimal assembly Givens instructions.

#include <immintrin.h>

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

#ifdef __SSE2__
    #define VECTOR_SIZE_2 2
    typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
    #define vall2(x) ((double2) _mm_set1_pd(x))
    #define vload2(v) ((double2) _mm_load_pd(v))
    #define vstore2(u, v) (_mm_store_pd(u, v))
    void apply_givens_SSE(const double S, const double C, double * X, double * Y) {
        double2 x = vload2(X);
        double2 y = vload2(Y);

        vstore2(X, C*x + S*y);
        vstore2(Y, C*y - S*x);
    }
#endif
#ifdef __AVX__
    #define VECTOR_SIZE_4 4
    typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
    #define vall4(x) ((double4) _mm256_set1_pd(x))
    #define vload4(v) ((double4) _mm256_load_pd(v))
    #define vstore4(u, v) (_mm256_store_pd(u, v))
    void apply_givens_AVX(const double S, const double C, double * X, double * Y) {
        double4 x = vload4(X);
        double4 y = vload4(Y);

        vstore4(X, C*x + S*y);
        vstore4(Y, C*y - S*x);
    }
#endif
#ifdef __AVX512F__
    #define VECTOR_SIZE_8 8
    typedef double double8 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
    #define vall8(x) ((double8) _mm512_set1_pd(x))
    #define vload8(v) ((double8) _mm512_load_pd(v))
    #define vstore8(u, v) (_mm512_store_pd(u, v))
    void apply_givens_AVX512(const double S, const double C, double * X, double * Y) {
        double8 x = vload8(X);
        double8 y = vload8(Y);

        vstore8(X, C*x + S*y);
        vstore8(Y, C*y - S*x);
    }
#endif

void gemv(char TRANS, int m, int n, double alpha, double * A, double * x, double beta, double * y) {
    double xj, t;
    if (TRANS == 'N') {
        for (int i = 0; i < m; i++)
            y[i] = beta*y[i];
        for (int j = 0; j < n; j++) {
            xj = alpha*x[j];
            for (int i = 0; i < m; i++)
                y[i] += A[i]*xj;
            A += m;
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++)
            y[i] = beta*y[i];
        for (int i = 0; i < n; i++) {
            t = 0.0;
            for (int j = 0; j < m; j++)
                t += A[j]*x[j];
            y[i] += alpha*t;
            A += m;
        }
    }
}
