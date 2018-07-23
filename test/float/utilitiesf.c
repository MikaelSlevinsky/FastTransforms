// Utility functions for testing.

#include "utilitiesf.h"

#define A(i,j) A[(i)+n*(j)]
#define B(i,j) B[(i)+n*(j)]
#define s(l,m) s[l+(m)*(2*n+1-(m))/2]
#define c(l,m) c[l+(m)*(2*n+1-(m))/2]

void printmat(char * MAT, float * A, int n, int m) {
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            printf("%s[%d][%d] = %17.16f\n", MAT, i, j, A(i,j));
}

float vecnorm_1arg(float * A, int n, int m) {
    float ret = 0.0;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            ret += pow(A(i,j), 2);
    return sqrt(ret);
}

float vecnorm_2arg(float * A, float * B, int n, int m) {
    float ret = 0.0;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            ret += pow(A(i,j)-B(i,j), 2);
    return sqrt(ret);
}

float vecnormInf_1arg(float * A, int n, int m) {
    float ret = 0.0, temp;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++) {
            temp = fabs(A(i,j));
            if (temp > ret)
                ret = temp;
        }
    return ret;
}

float vecnormInf_2arg(float * A, float * B, int n, int m) {
    float ret = 0.0, temp;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++) {
            temp = fabs(A(i,j)-B(i,j));
            if (temp > ret)
                ret = temp;
        }
    return ret;
}

float rotnorm(const RotationPlan * RP) {
    float * s = RP->s, * c = RP->c;
    float ret = 0.0;
    int n = RP->n;
    for (int m = 0; m < n; m++)
        for (int l = 0; l < n-m; l++)
            ret += pow(hypot(s(l,m), c(l,m)) - 1.0, 2);
    return sqrt(ret);
}

float * sphones(int n, int m) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-2*i; j++)
            A(i,j) = 1.0;
    return A;
}

float * sphrand(int n, int m) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-2*i; j++)
            A(i,j) = 2.0*(((float) rand())/RAND_MAX)-1.0;
    return A;
}

float * triones(int n, int m) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-i; j++)
            A(i,j) = 1.0;
    return A;
}

float * trirand(int n, int m) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-i; j++)
            A(i,j) = 2.0*(((float) rand())/RAND_MAX)-1.0;
    return A;
}

float * diskones(int n, int m) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-4*i; j++)
            A(i,j) = 1.0;
    return A;
}

float * diskrand(int n, int m) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-4*i; j++)
            A(i,j) = 2.0*(((float) rand())/RAND_MAX)-1.0;
    return A;
}

float * spinsphones(int n, int m, int s) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n-s; i++)
        for (int j = 0; j < m-2*i; j++)
            A(i,j) = 1.0;
    return A;
}

float * spinsphrand(int n, int m, int s) {
    float * A = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n-s; i++)
        for (int j = 0; j < m-2*i; j++)
            A(i,j) = 2.0*(((float) rand())/RAND_MAX)-1.0;
    return A;
}

float * copyA(float * A, int n, int m) {
    float * B = (float *) calloc(n * m, sizeof(float));
    for (int i = 0; i < n*m; i++)
        B[i] = A[i];
    return B;
}

float elapsed(struct timeval * start, struct timeval * end, int N) {
    return ((end->tv_sec  - start->tv_sec) * 1000000u + end->tv_usec - start->tv_usec) / (1.e6 * N);
}
