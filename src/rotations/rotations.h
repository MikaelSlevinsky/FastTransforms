#ifndef FTROTATIONS_H
#define FTROTATIONS_H

#include "ftinternal.h"

#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

#define KERNEL_TRI_HI2LO(T, VT, VS, L, VLOAD, VSTORE, APPLY_GIVENS)            \
int n = RP->n, row, col;                                                       \
T ts, tc;                                                                      \
VT X[2*L], T1, T2;                                                             \
for (int s = 1; s < VS; s++)                                                   \
    kernel_tri_hi2lo_default(RP, m2, m2+s, A+s, S);                            \
int j = m2-1;                                                                  \
for (; j >= m1+L-1; j -= L) {                                                  \
    int k = n-2-j;                                                             \
    for (; k >= L-1; k -= L) {                                                 \
        for (int l = 0; l < 2*L; l++)                                          \
            X[l] = VLOAD(A+(k+1-L+l)*S);                                       \
        for (int lj = 0; lj < L; lj++) {                                       \
            for (int lk = 0; lk < L; lk++) {                                   \
                row = k-lk+lj;                                                 \
                col = j-lj;                                                    \
                T1 = X[L-1-lk+lj];                                             \
                T2 = X[L-lk+lj];                                               \
                ts = RP->s(row, col);                                          \
                tc = RP->c(row, col);                                          \
                X[L-1-lk+lj] = tc*T1 + ts*T2;                                  \
                X[L-lk+lj] = tc*T2 - ts*T1;                                    \
            }                                                                  \
        }                                                                      \
        for (int l = 0; l < 2*L; l++)                                          \
            VSTORE(A+(k+1-L+l)*S, X[l]);                                       \
    }                                                                          \
    for (int lj = 0; lj < L; lj++) {                                           \
        for (int lk = k+lj; lk >= 0; lk--) {                                   \
            row = lk;                                                          \
            col = j-lj;                                                        \
            APPLY_GIVENS(RP->s(row, col), RP->c(row, col), A+row*S, A+(row+1)*S); \
        }                                                                      \
    }                                                                          \
}                                                                              \
for (; j >= m1; j--) {                                                         \
    for (int k = n-2-j; k >= 0; k--) {                                         \
        APPLY_GIVENS(RP->s(k, j), RP->c(k, j), A+k*S, A+(k+1)*S);              \
    }                                                                          \
}

#define KERNEL_TRI_LO2HI(T, VT, VS, L, VLOAD, VSTORE, APPLY_GIVENS_T)          \
int n = RP->n, row, col;                                                       \
T ts, tc;                                                                      \
VT X[2*L], T1, T2;                                                             \
int j = m1;                                                                    \
for (; j < m2-L+1; j += L) {                                                   \
    int k = L+(n-L-1-j)%L;                                                     \
    for (int lj = 0; lj < L; lj++) {                                           \
        for (int lk = 0; lk < k-lj; lk++) {                                    \
            row = lk;                                                          \
            col = j+lj;                                                        \
            APPLY_GIVENS_T(RP->s(row, col), RP->c(row, col), A+row*S, A+(row+1)*S); \
        }                                                                      \
    }                                                                          \
    for (; k <= n-L-1-j; k += L) {                                             \
        for (int l = 0; l < 2*L; l++)                                          \
            X[l] = VLOAD(A+(k+1-L+l)*S);                                       \
        for (int lj = 0; lj < L; lj++) {                                       \
            for (int lk = 0; lk < L; lk++) {                                   \
                row = k+lk-lj;                                                 \
                col = j+lj;                                                    \
                T1 = X[L-1+lk-lj];                                             \
                T2 = X[L+lk-lj];                                               \
                ts = RP->s(row, col);                                          \
                tc = RP->c(row, col);                                          \
                X[L-1+lk-lj] = tc*T1 - ts*T2;                                  \
                X[L+lk-lj] = tc*T2 + ts*T1;                                    \
            }                                                                  \
        }                                                                      \
        for (int l = 0; l < 2*L; l++)                                          \
            VSTORE(A+(k+1-L+l)*S, X[l]);                                       \
    }                                                                          \
}                                                                              \
for (; j < m2; j++) {                                                          \
    for (int k = 0; k <= n-2-j; k++) {                                         \
        APPLY_GIVENS_T(RP->s(k, j), RP->c(k, j), A+k*S, A+(k+1)*S);            \
    }                                                                          \
}                                                                              \
for (int s = 1; s < VS; s++)                                                   \
    kernel_tri_lo2hi_default(RP, m2, m2+s, A+s, S);

#endif // FTROTATIONS_H
