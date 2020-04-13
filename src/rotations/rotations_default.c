#include "rotations.h"

static inline void apply_givens(const double S, const double C, double * X, double * Y) {
    double x = C*X[0] + S*Y[0];
    double y = C*Y[0] - S*X[0];

    X[0] = x;
    Y[0] = y;
}

static inline void apply_givens_t(const double S, const double C, double * X, double * Y) {
    double x = C*X[0] - S*Y[0];
    double y = C*Y[0] + S*X[0];

    X[0] = x;
    Y[0] = y;
}

void kernel_sph_hi2lo_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    int n = RP->n;
    for (int j = m2-2; j >= m1; j -= 2)
        for (int k = n-3-j; k >= 0; k--)
            apply_givens(RP->s(k, j), RP->c(k, j), A+k*S, A+(k+2)*S);
}

void kernel_sph_lo2hi_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    int n = RP->n;
    for (int j = m1; j < m2; j += 2)
        for (int k = 0; k <= n-3-j; k++)
            apply_givens_t(RP->s(k, j), RP->c(k, j), A+k*S, A+(k+2)*S);
}

void kernel_tri_hi2lo_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    int n = RP->n;
    for (int j = m2-1; j >= m1; j--)
        for (int k = n-2-j; k >= 0; k--)
            apply_givens(RP->s(k, j), RP->c(k, j), A+k*S, A+(k+1)*S);
}

void kernel_tri_lo2hi_default(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S) {
    int n = RP->n;
    for (int j = m1; j < m2; j++)
        for (int k = 0; k <= n-2-j; k++)
            apply_givens_t(RP->s(k, j), RP->c(k, j), A+k*S, A+(k+1)*S);
}
