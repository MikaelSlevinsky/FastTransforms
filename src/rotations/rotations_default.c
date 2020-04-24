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

static inline void apply_givensc(const double S, const double C, ft_complex * X, ft_complex * Y) {
    double x = C*X[0][0] + S*Y[0][0];
    double y = C*Y[0][0] - S*X[0][0];
    X[0][0] = x;
    Y[0][0] = y;
    x = C*X[0][1] + S*Y[0][1];
    y = C*Y[0][1] - S*X[0][1];
    X[0][1] = x;
    Y[0][1] = y;
}

static inline void apply_givens_tc(const double S, const double C, ft_complex * X, ft_complex * Y) {
    double x = C*X[0][0] - S*Y[0][0];
    double y = C*Y[0][0] + S*X[0][0];
    X[0][0] = x;
    Y[0][0] = y;
    x = C*X[0][1] - S*Y[0][1];
    y = C*Y[0][1] + S*X[0][1];
    X[0][1] = x;
    Y[0][1] = y;
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

void kernel_spinsph_hi2lo_default(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    if (m*s >= 0)
        for (int j = MIN(am, as)-1; j >= 0; j--)
            for (int k = n-2-abs(am-as)-j; k >= 0; k--)
                apply_givensc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
    else
        for (int j = MIN(am, as)-1; j >= 0; j--)
            for (int k = n-2-abs(am-as)-j; k >= 0; k--)
                apply_givens_tc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
    for (int j = abs(am-as)-2; j >= (am+as)%2; j -= 2)
        for (int k = n-3-j; k >= 0; k--)
            apply_givensc(SRP->s1(k, j), SRP->c1(k, j), A+k*S, A+(k+2)*S);
}

void kernel_spinsph_lo2hi_default(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S) {
    int n = SRP->n, s = SRP->s;
    int as = abs(s), am = abs(m);
    for (int j = (am+as)%2; j <= abs(am-as)-2; j += 2)
        for (int k = 0; k <= n-3-j; k++)
            apply_givens_tc(SRP->s1(k, j), SRP->c1(k, j), A+k*S, A+(k+2)*S);
    if (m*s >= 0)
        for (int j = 0; j < MIN(am, as); j++)
            for (int k = 0; k <= n-2-abs(am-as)-j; k++)
                apply_givens_tc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
    else
        for (int j = 0; j < MIN(am, as); j++)
            for (int k = 0; k <= n-2-abs(am-as)-j; k++)
                apply_givensc(SRP->s2(k, j, abs(am-as)), SRP->c2(k, j, abs(am-as)), A+k*S, A+(k+1)*S);
}
