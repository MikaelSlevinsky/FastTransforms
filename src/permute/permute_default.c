#include "permute.h"

void swap_warp_default(double * A, double * B, const int N) {
    double temp;
    for (int i = 0; i < N; i++) {
        temp = A[i];
        A[i] = B[i];
        B[i] = temp;
    }
}

void swap_warp_defaultf(float * A, float * B, const int N) {
    float temp;
    for (int i = 0; i < N; i++) {
        temp = A[i];
        A[i] = B[i];
        B[i] = temp;
    }
}
