// Computational routines for the harmonic polynomial connection problem.

#include "rotations.h"

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

RotationPlan * plan_rotsphere(const int n) {
    double * s = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double * c = (double *) malloc(n*(n+1)/2 * sizeof(double));
    double nums, numc, den;
    for (int l = 0; l < n; l++)
        for (int m = 0; m < n-l; m++) {
            nums = (l+1)*(l+2);
            numc = (2*m+2)*(2*l+2*m+5);
            den = (l+2*m+3)*(l+2*m+4);
            s(l, m) = sqrt(nums/den);
            c(l, m) = sqrt(numc/den);
        }
    RotationPlan * RP = malloc(sizeof(RotationPlan));
    RP->s = s;
    RP->c = c;
    RP->n = n;
    return RP;
}

// Convert a single vector of spherical harmonics of order m to 0/1.

void kernel1_sph_hi2lo(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_1x1(RP->s, RP->c, n, l, j, A);
}

// Convert a single vector of spherical harmonics of order 0/1 to m.

void kernel1_sph_lo2hi(const RotationPlan * RP, const int m, double * A) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_1x1(RP->s, RP->c, n, l, j, A);
}

// Convert a pair of vectors of spherical harmonics of order m to 0/1.

void kernel2_sph_hi2lo(const RotationPlan * RP, const int m, double * A, double * B) {
    int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2)
        for (int l = n-3-j; l >= 0; l--)
            apply_givens_2x1(RP->s, RP->c, n, l, j, A, B);
}

// Convert a pair of vectors of spherical harmonics of order 0/1 to m.

void kernel2_sph_lo2hi(const RotationPlan * RP, const int m, double * A, double * B) {
    int n = RP->n;
    for (int j = m%2; j < m-1; j += 2)
        for (int l = 0; l <= n-3-j; l++)
            apply_givens_t_2x1(RP->s, RP->c, n, l, j, A, B);
}

// Convert a pair of vectors of spherical harmonics of order m to 0/1.

void kernel2x4_sph_hi2lo(const RotationPlan * RP, const int m, double * A, double * B) {
    const double * s = RP->s, * c = RP->c;
    const int n = RP->n;
    for (int j = m-2; j >= 0; j -= 2) {
        if ((n-j-2)%4 == 0) {
            for (int l = n-5-j; l > 0; l -= 4) {
                apply_givens_2x2(s, c, n, l  , j, A, B);
                apply_givens_2x2(s, c, n, l-1, j, A, B);
            }
        }
        else if ((n-j-2)%4 == 1) {
            apply_givens_2x1(s, c, n, n-3-j, j, A, B);
            for (int l = n-6-j; l > 0; l -= 4) {
                apply_givens_2x2(s, c, n, l  , j, A, B);
                apply_givens_2x2(s, c, n, l-1, j, A, B);
            }
        }
        else if ((n-j-2)%4 == 2) {
            apply_givens_2x1(s, c, n, n-3-j, j, A, B);
            apply_givens_2x1(s, c, n, n-4-j, j, A, B);
            for (int l = n-7-j; l > 0; l -= 4) {
                apply_givens_2x2(s, c, n, l  , j, A, B);
                apply_givens_2x2(s, c, n, l-1, j, A, B);
            }
        }
        else if ((n-j-2)%4 == 3) {
            apply_givens_2x1(s, c, n, n-3-j, j, A, B);
            apply_givens_2x1(s, c, n, n-4-j, j, A, B);
            apply_givens_2x1(s, c, n, n-5-j, j, A, B);
            for (int l = n-8-j; l > 0; l -= 4) {
                apply_givens_2x2(s, c, n, l  , j, A, B);
                apply_givens_2x2(s, c, n, l-1, j, A, B);
            }
        }
    }
}

// Convert a pair of vectors of spherical harmonics of order 0/1 to m.

void kernel2x4_sph_lo2hi(const RotationPlan * RP, const int m, double * A, double * B) {
    const double * s = RP->s, * c = RP->c;
    const int n = RP->n;
    for (int j = m%2; j < m-1; j += 2) {
        if ((n-j-2)%4 == 0) {
            for (int l = 1; l <= n-5-j; l += 4) {
                apply_givens_t_2x2(s, c, n, l-1, j, A, B);
                apply_givens_t_2x2(s, c, n, l  , j, A, B);
            }
        }
        else if ((n-j-2)%4 == 1) {
            for (int l = 1; l <= n-6-j; l += 4) {
                apply_givens_t_2x2(s, c, n, l-1, j, A, B);
                apply_givens_t_2x2(s, c, n, l  , j, A, B);
            }
            apply_givens_t_2x1(s, c, n, n-3-j, j, A, B);
        }
        else if ((n-j-2)%4 == 2) {
            for (int l = 1; l <= n-7-j; l += 4) {
                apply_givens_t_2x2(s, c, n, l-1, j, A, B);
                apply_givens_t_2x2(s, c, n, l  , j, A, B);
            }
            apply_givens_t_2x1(s, c, n, n-4-j, j, A, B);
            apply_givens_t_2x1(s, c, n, n-3-j, j, A, B);
        }
        else if ((n-j-2)%4 == 3) {
            for (int l = 1; l <= n-8-j; l += 4) {
                apply_givens_t_2x2(s, c, n, l-1, j, A, B);
                apply_givens_t_2x2(s, c, n, l  , j, A, B);
            }
            apply_givens_t_2x1(s, c, n, n-5-j, j, A, B);
            apply_givens_t_2x1(s, c, n, n-4-j, j, A, B);
            apply_givens_t_2x1(s, c, n, n-3-j, j, A, B);
        }
    }
}

static inline void apply_givens_1x1(const double * s, const double * c, const int n, const int l, const int m, double * A) {
    register double s1, c1;
    register double a1, a2;
    s1 = s(l, m);
    c1 = c(l, m);

    a1 = A[l  ];
    a2 = A[l+2];

    A[l  ] = c1*a1 + s1*a2;
    A[l+2] = c1*a2 - s1*a1;
}

static inline void apply_givens_t_1x1(const double * s, const double * c, const int n, const int l, const int m, double * A) {
    register double s1, c1;
    register double a1, a2;
    s1 = s(l, m);
    c1 = c(l, m);

    a1 = A[l  ];
    a2 = A[l+2];

    A[l  ] = c1*a1 - s1*a2;
    A[l+2] = c1*a2 + s1*a1;
}

static inline void apply_givens_2x1(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B) {
    register double s1, c1;
    register double a1, a2;
    s1 = s(l, m);
    c1 = c(l, m);

    a1 = A[l  ];
    a2 = A[l+2];

    A[l  ] = c1*a1 + s1*a2;
    A[l+2] = c1*a2 - s1*a1;

    a1 = B[l  ];
    a2 = B[l+2];

    B[l  ] = c1*a1 + s1*a2;
    B[l+2] = c1*a2 - s1*a1;
}

static inline void apply_givens_t_2x1(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B) {
    register double s1, c1;
    register double a1, a2;
    s1 = s(l, m);
    c1 = c(l, m);

    a1 = A[l  ];
    a2 = A[l+2];

    A[l  ] = c1*a1 - s1*a2;
    A[l+2] = c1*a2 + s1*a1;

    a1 = B[l  ];
    a2 = B[l+2];

    B[l  ] = c1*a1 - s1*a2;
    B[l+2] = c1*a2 + s1*a1;
}

static inline void apply_givens_2x2(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B) {
    register double s1, s2, c1, c2;
    register double a1, a2, a3, t;
    s1 = s(l  , m);
    c1 = c(l  , m);
    s2 = s(l+2, m);
    c2 = c(l+2, m);

    a1 = A[l  ];
    a2 = A[l+2];
    a3 = A[l+4];

    t = c2*a2 + s2*a3;

    A[l  ] = c1*a1 + s1*t ;
    A[l+2] = c1*t  - s1*a1;
    A[l+4] = c2*a3 - s2*a2;

    a1 = B[l];
    a2 = B[l+2];
    a3 = B[l+4];

    t = c2*a2 + s2*a3;

    B[l  ] = c1*a1 + s1*t ;
    B[l+2] = c1*t  - s1*a1;
    B[l+4] = c2*a3 - s2*a2;
}

static inline void apply_givens_t_2x2(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B) {
    register double s1, s2, c1, c2;
    register double a1, a2, a3, t;
    s1 = s(l  , m);
    c1 = c(l  , m);
    s2 = s(l+2, m);
    c2 = c(l+2, m);

    a1 = A[l  ];
    a2 = A[l+2];
    a3 = A[l+4];

    t = c1*a2 + s1*a1;

    A[l  ] = c1*a1 - s1*a2;
    A[l+2] = c2*t  - s2*a3;
    A[l+4] = c2*a3 + s2*t ;

    a1 = B[l];
    a2 = B[l+2];
    a3 = B[l+4];

    t = c1*a2 + s1*a1;

    B[l  ] = c1*a1 - s1*a2;
    B[l+2] = c2*t  - s2*a3;
    B[l+4] = c2*a3 + s2*t ;
}
