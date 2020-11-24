// Utility functions for testing.

#include "ftutilities.h"

#define A(i,j) A[(i)+n*(j)]
#define B(i,j) B[(i)+n*(j)]

void printmat(char * MAT, char * FMT, double * A, int n, int m) {
    printf("%s = \n", MAT);
    if (n > 0 && m > 0) {
        if (signbit(A(0,0))) {printf("[");}
        else {printf("[ ");}
        printf(FMT, A(0,0));
        for (int j = 1; j < m; j++) {
            if (signbit(A(0,j))) {printf("  ");}
            else {printf("   ");}
            printf(FMT, A(0,j));
        }
        for (int i = 1; i < n-1; i++) {
            printf("\n");
            if (signbit(A(i,0))) {printf(" ");}
            else {printf("  ");}
            printf(FMT, A(i,0));
            for (int j = 1; j < m; j++) {
                if (signbit(A(i,j))) {printf("  ");}
                else {printf("   ");}
                printf(FMT, A(i,j));
            }
        }
        if (n > 1) {
            printf("\n");
            if (signbit(A(n-1,0))) {printf(" ");}
            else {printf("  ");}
            printf(FMT, A(n-1,0));
            for (int j = 1; j < m; j++) {
                if (signbit(A(n-1,j))) {printf("  ");}
                else {printf("   ");}
                printf(FMT, A(n-1,j));
            }
        }
        printf("]\n");
    }
}

void print_summary_size(size_t i) {
    if (i < 1024.0)
        printf("%20zu B\n", i);
    else if (i < 1048576.0)
        printf("%18.3f KiB\n", i/1024.0);
    else if (i < 1073741824.0)
        printf("%18.3f MiB\n", i/1048576.0);
    else if (i < 1099511627776.0)
        printf("%18.3f GiB\n", i/1073741824.0);
    else if (i < 1.125899906842624e15)
        printf("%18.3f TiB\n", i/1099511627776.0);
    else
        printf("%18.14e B\n", (double) i);
}

double * copymat(double * A, int n, int m) {
    double * B = calloc(n*m, sizeof(double));
    for (int i = 0; i < n*m; i++)
        B[i] = A[i];
    return B;
}

double * sphones(int n, int m) {
    double * A  = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-2*i; j++)
            A(i,j) = 1.0;
    return A;
}

double * sphrand(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-2*i; j++)
            A(i,j) = 2.0*(((double) rand())/RAND_MAX)-1.0;
    return A;
}

double * triones(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-i; j++)
            A(i,j) = 1.0;
    return A;
}

double * trirand(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-i; j++)
            A(i,j) = 2.0*(((double) rand())/RAND_MAX)-1.0;
    return A;
}

double * diskones(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-4*i; j++)
            A(i,j) = 1.0;
    return A;
}

double * diskrand(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-4*i; j++)
            A(i,j) = 2.0*(((double) rand())/RAND_MAX)-1.0;
    return A;
}

double * rectdiskones(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-i; j++)
            A(i,j) = 1.0;
    return A;
}

double * rectdiskrand(int n, int m) {
    double * A = calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m-i; j++)
            A(i,j) = 2.0*(((double) rand())/RAND_MAX)-1.0;
    return A;
}

double * tetones(int n, int l, int m) {
    double * A = calloc(n * l * m, sizeof(double));
    for (int k = 0; k < m; k++)
        for (int j = 0; j < l-k; j++)
            for (int i = 0; i < n-j-k; i++)
                A[i+l*(j+n*k)] = 1.0;
    return A;
}

double * tetrand(int n, int l, int m) {
    double * A = calloc(n * l * m, sizeof(double));
    for (int k = 0; k < m; k++)
        for (int j = 0; j < l-k; j++)
            for (int i = 0; i < n-j-k; i++)
                A[i+l*(j+n*k)] = 2.0*(((double) rand())/RAND_MAX)-1.0;
    return A;
}

ft_complex * spinsphones(int n, int m, int s) {
    ft_complex * A = calloc(n * m, sizeof(ft_complex));
    for (int i = 0; i < n-abs(s); i++)
        for (int j = 0; j < m-2*i; j++) {
            A[i+n*j][0] = 1.0;
            A[i+n*j][1] = 1.0;
        }
    return A;
}

ft_complex * spinsphrand(int n, int m, int s) {
    ft_complex * A = calloc(n * m, sizeof(ft_complex));
    for (int i = 0; i < n-abs(s); i++)
        for (int j = 0; j < m-2*i; j++) {
            A[i+n*j][0] = 2.0*(((double) rand())/RAND_MAX)-1.0;
            A[i+n*j][1] = 2.0*(((double) rand())/RAND_MAX)-1.0;
        }
    return A;
}

double elapsed(struct timeval * start, struct timeval * end, int N) {
    return ((end->tv_sec  - start->tv_sec) * 1000000u + end->tv_usec - start->tv_usec) / (1.e6 * N);
}

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "ftutilities_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "ftutilities_source.c"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "ftutilities_source.c"
#undef FLT
#undef X
#undef Y

#if defined(FT_QUADMATH)
    #define FLT quadruple
    #define X(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, q)
    #include "ftutilities_source.c"
    #undef FLT
    #undef X
    #undef Y
#endif
