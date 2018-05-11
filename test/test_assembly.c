// Staging ground for determining the minimal assembly Givens instructions.

//#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
//#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

#include <immintrin.h>

#define VECTOR_SIZE_2 2
typedef double double2 __attribute__ ((vector_size (VECTOR_SIZE_2*8)));
#define vall2(x) ((double2) _mm_set1_pd(x))
#define vload2(v) ((double2) _mm_loadu_pd(v))
#define vstore2(u, v) (_mm_storeu_pd(u, v))

#define VECTOR_SIZE_4 4
typedef double double4 __attribute__ ((vector_size (VECTOR_SIZE_4*8)));
#define vall4(x) ((double4) _mm256_set1_pd(x))
#define vload4(v) ((double4) _mm256_loadu_pd(v))
#define vstore4(u, v) (_mm256_storeu_pd(u, v))

#define VECTOR_SIZE_8 8
typedef double double8 __attribute__ ((vector_size (VECTOR_SIZE_8*8)));
#define vall8(x) ((double8) _mm512_set1_pd(x))
#define vload8(v) ((double8) _mm512_loadu_pd(v))
#define vstore8(u, v) (_mm512_storeu_pd(u, v))

void apply_givens(const int inc, const double s, const double c, const int n, const int l, const int m, double * A) {
    double s1 = s;//(l, m);
    double c1 = c;//(l, m);

    double a1 = A[l];
    double a2 = A[l+inc];

    A[l    ] = c1*a1 + s1*a2;
    A[l+inc] = c1*a2 - s1*a1;

}

void apply_givens_SSE(const int inc, const double s, const double c, const int n, const int l, const int m, double * A) {
    double2 s1 = vall2(s);//(l, m));
    double2 c1 = vall2(c);//(l, m));

    double2 a1 = vload2(A+2*l      );
    double2 a2 = vload2(A+2*(l+inc));

    vstore2(A+2*l,       c1*a1 + s1*a2);
    vstore2(A+2*(l+inc), c1*a2 - s1*a1);
}

void apply_givens_AVX(const int inc, const double s, const double c, const int n, const int l, const int m, double * A) {
    double4 s1 = vall4(s);//(l, m));
    double4 c1 = vall4(c);//(l, m));

    double4 a1 = vload4(A+4*l      );
    double4 a2 = vload4(A+4*(l+inc));

    vstore4(A+4*l,       c1*a1 + s1*a2);
    vstore4(A+4*(l+inc), c1*a2 - s1*a1);
}

void apply_givens_AVX512(const int inc, const double s, const double c, const int n, const int l, const int m, double * A) {
    double8 s1 = vall8(s);//(l, m));
    double8 c1 = vall8(c);//(l, m));

    double8 a1 = vload8(A+8*l      );
    double8 a2 = vload8(A+8*(l+inc));

    vstore8(A+8*l,       c1*a1 + s1*a2);
    vstore8(A+8*(l+inc), c1*a2 - s1*a1);
}
