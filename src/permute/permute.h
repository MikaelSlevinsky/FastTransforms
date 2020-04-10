#ifndef FTPERMUTE_H
#define FTPERMUTE_H

#include "ftinternal.h"

#define SWAP_WARP_KERNEL(T, VT, S, L, VLOAD, VSTORE)                           \
int i = 0;                                                                     \
for (; i < N+1-S*L; i += S*L) {                                                \
    VT TEMPA[L], TEMPB[L];                                                     \
    for (int l = 0; l < L; l++) {                                              \
        TEMPA[l] = VLOAD(A+i+S*l);                                             \
        TEMPB[l] = VLOAD(B+i+S*l);                                             \
        VSTORE(A+i+S*l, TEMPB[l]);                                             \
        VSTORE(B+i+S*l, TEMPA[l]);                                             \
    }                                                                          \
}                                                                              \
for (; i < N; i++) {                                                           \
    T temp = A[i];                                                             \
    A[i] = B[i];                                                               \
    B[i] = temp;                                                               \
}

#endif // FTPERMUTE_H
