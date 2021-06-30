#ifndef FTRECURRENCE_H
#define FTRECURRENCE_H

#include "ftinternal.h"

#define HORNER_KERNEL(T, VT, S, L, VLOAD, VSTORE, VMULADD, VALL)               \
if (n < 1) {                                                                   \
    for (int j = 0; j < m; j++)                                                \
        f[j] = 0;                                                              \
    return;                                                                    \
}                                                                              \
int j = 0;                                                                     \
for (; j < m+1-S*L; j += S*L) {                                                \
    VT bk[L] = {0};                                                            \
    VT X[L];                                                                   \
    for (int l = 0; l < L; l++)                                                \
        X[l] = VLOAD(x+j+S*l);                                                 \
    for (int k = n-1; k >= 0; k--) {                                           \
        for (int l = 0; l < L; l++)                                            \
            bk[l] = VMULADD(X[l], bk[l], VALL(c[k*incc]));                     \
    }                                                                          \
    for (int l = 0; l < L; l++)                                                \
        VSTORE(f+j+S*l, bk[l]);                                                \
}                                                                              \
for (; j < m; j++) {                                                           \
    T bk = 0;                                                                  \
    for (int k = n-1; k >= 0; k--)                                             \
        bk = x[j]*bk + c[k*incc];                                              \
    f[j] = bk;                                                                 \
}

#define CLENSHAW_KERNEL(T, VT, S, L, VLOAD, VSTORE, VMULADD, VALL)             \
if (n < 1) {                                                                   \
    for (int j = 0; j < m; j++)                                                \
        f[j] = 0;                                                              \
    return;                                                                    \
}                                                                              \
int j = 0;                                                                     \
for (; j < m+1-S*L; j += S*L) {                                                \
    T TWO = 2;                                                                 \
    VT bk[3*L] = {0};                                                          \
    VT X[L];                                                                   \
    for (int l = 0; l < L; l++)                                                \
        X[l] = VALL(TWO)*VLOAD(x+j+S*l);                                       \
    for (int k = n-1; k >= 1; k--) {                                           \
        for (int l = 0; l < L; l++) {                                          \
            bk[3*l] = VMULADD(X[l], bk[3*l+1], VALL(c[k*incc]) - bk[3*l+2]);   \
            bk[3*l+2] = bk[3*l+1];                                             \
            bk[3*l+1] = bk[3*l];                                               \
        }                                                                      \
    }                                                                          \
    for (int l = 0; l < L; l++)                                                \
        VSTORE(f+j+S*l, X[l]/VALL(TWO)*bk[3*l+1] + VALL(c[0]) - bk[3*l+2]);    \
}                                                                              \
for (; j < m; j++) {                                                           \
    T bk = 0;                                                                  \
    T bk1 = 0;                                                                 \
    T bk2 = 0;                                                                 \
    T X = 2*x[j];                                                              \
    for (int k = n-1; k >= 1; k--) {                                           \
        bk = X*bk1 + c[k*incc] - bk2;                                          \
        bk2 = bk1;                                                             \
        bk1 = bk;                                                              \
    }                                                                          \
    f[j] = X/2*bk1 + c[0] - bk2;                                               \
}

#define ORTHOGONAL_POLYNOMIAL_CLENSHAW_KERNEL(T, VT, S, L, VLOAD, VSTORE, VMULADD, VMULSUB, VALL) \
if (n < 1) {                                                                   \
    for (int j = 0; j < m; j++)                                                \
        f[j] = 0;                                                              \
    return;                                                                    \
}                                                                              \
int j = 0;                                                                     \
for (; j < m+1-S*L; j += S*L) {                                                \
    VT bk[3*L] = {0};                                                          \
    VT X[L];                                                                   \
    for (int l = 0; l < L; l++)                                                \
        X[l] = VLOAD(x+j+S*l);                                                 \
    for (int k = n-1; k >= 0; k--) {                                           \
        for (int l = 0; l < L; l++) {                                          \
            bk[3*l] = VMULSUB(VMULADD(VALL(A[k]), X[l], VALL(B[k])), bk[3*l+1], VMULSUB(VALL(C[k+1]), bk[3*l+2], VALL(c[k*incc]))); \
            bk[3*l+2] = bk[3*l+1];                                             \
            bk[3*l+1] = bk[3*l];                                               \
        }                                                                      \
    }                                                                          \
    for (int l = 0; l < L; l++)                                                \
        VSTORE(f+j+S*l, VLOAD(phi0+j+S*l)*bk[3*l]);                            \
}                                                                              \
for (; j < m; j++) {                                                           \
    T bk = 0;                                                                  \
    T bk1 = 0;                                                                 \
    T bk2 = 0;                                                                 \
    T X = x[j];                                                                \
    for (int k = n-1; k >= 0; k--) {                                           \
        bk = (A[k]*X+B[k])*bk1 - C[k+1]*bk2 + c[k*incc];                       \
        bk2 = bk1;                                                             \
        bk1 = bk;                                                              \
    }                                                                          \
    f[j] = phi0[j]*bk;                                                         \
}

#define EIGEN_EVAL_KERNEL(T, VT, S, L, VLOAD, VSTORE, VMULADD, VALL, VSQRT, VMOVEMASK, HUGE, SSQRT) \
if (n < 1) {                                                                   \
    for (int j = 0; j < m; j++)                                                \
        f[j] = 0;                                                              \
    return;                                                                    \
}                                                                              \
int j = 0;                                                                     \
for (; j < m+1-S*L; j += S*L) {                                                \
    T ONE = 1;                                                                 \
    VT vkm1[L];                                                                \
    VT vk[L];                                                                  \
    VT vkp1[L] = {0};                                                          \
    VT nrm[L];                                                                 \
    VT sum[L];                                                                 \
    VT X[L];                                                                   \
    for (int l = 0; l < L; l++) {                                              \
        vkm1[l] = VALL(ONE);                                                   \
        vk[l] = VALL(ONE);                                                     \
        nrm[l] = VALL(ONE);                                                    \
        sum[l] = VALL(c[(n-1)*incc]);                                          \
        X[l] = VLOAD(x+j+S*l);                                                 \
    }                                                                          \
    for (int k = n-1; k > 0; k--) {                                            \
        for (int l = 0; l < L; l++) {                                          \
            vkm1[l] = VALL(A[k])*VMULADD(X[l]+VALL(B[k]), vk[l], VALL(C[k])*vkp1[l]); \
            vkp1[l] = vk[l];                                                   \
            vk[l] = vkm1[l];                                                   \
            nrm[l] = VMULADD(vkm1[l], vkm1[l], nrm[l]);                        \
            sum[l] = VMULADD(vkm1[l], VALL(c[(k-1)*incc]), sum[l]);            \
            if (VMOVEMASK(nrm[l] > VALL(HUGE)) != 0) {                         \
                nrm[l] = VALL(ONE)/VSQRT(nrm[l]);                              \
                vkp1[l] = nrm[l]*vkp1[l];                                      \
                vk[l] = nrm[l]*vk[l];                                          \
                vkm1[l] = nrm[l]*vkm1[l];                                      \
                sum[l] = nrm[l]*sum[l];                                        \
                nrm[l] = VALL(ONE);                                            \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    for (int l = 0; l < L; l++) {                                              \
        nrm[l] = VALL(ONE)/VSQRT(nrm[l]);                                      \
        VSTORE(f+j+S*l, nrm[l]*sum[l]);                                        \
    }                                                                          \
    T svkm1[S*L];                                                              \
    for (int l = 0; l < L; l++)                                                \
        VSTORE(&svkm1[0]+S*l, vkm1[l]);                                        \
    for (int ll = 0; ll < S*L; ll++)                                           \
        f[j+ll] = (sign*svkm1[ll] < 0) ? -f[j+ll] : f[j+ll];                   \
}                                                                              \
for (; j < m; j++) {                                                           \
    T vkm1 = 1;                                                                \
    T vk = 1;                                                                  \
    T vkp1 = 0;                                                                \
    T nrm = 1;                                                                 \
    T X = x[j];                                                                \
    T sum = c[(n-1)*incc];                                                     \
    for (int k = n-1; k > 0; k--) {                                            \
        vkm1 = A[k]*((X+B[k])*vk + C[k]*vkp1);                                 \
        vkp1 = vk;                                                             \
        vk = vkm1;                                                             \
        nrm += vkm1*vkm1;                                                      \
        sum += vkm1*c[(k-1)*incc];                                             \
        if (nrm > HUGE) {                                                      \
            nrm = 1/SSQRT(nrm);                                                \
            vkp1 *= nrm;                                                       \
            vk *= nrm;                                                         \
            vkm1 *= nrm;                                                       \
            sum *= nrm;                                                        \
            nrm = 1;                                                           \
        }                                                                      \
    }                                                                          \
    nrm = (sign*vkm1 < 0) ? -1/SSQRT(nrm) : 1/SSQRT(nrm);                      \
    f[j] = nrm*sum;                                                            \
}

#endif // FTRECURRENCE_H
