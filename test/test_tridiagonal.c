#include "fasttransforms.h"
#include "ftutilities.h"

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "test_tridiagonal_source.c"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "test_tridiagonal_source.c"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "test_tridiagonal_source.c"
#undef FLT
#undef X
#undef Y

#define FLT quadruple
#define X(name) FT_CONCAT(ft_, name, q)
#define Y(name) FT_CONCAT(, name, q)
#include "test_tridiagonal_source.c"
#undef FLT
#undef X
#undef Y

void symmetric_tridiagonal_printmat(char * MAT, char * FMT, ft_symmetric_tridiagonal * A);
void bidiagonal_printmat(char * MAT, char * FMT, ft_bidiagonal * B);

int main(void) {
    int checksum = 0;
    printf("\nTesting methods for bidiagonal and symmetric tridiagonal matrices.\n");
    printf("\n\tSingle precision.\n\n");
    test_tridiagonalf(&checksum);
    printf("\n\tDouble precision.\n\n");
    test_tridiagonal(&checksum);
    printf("\n\tLong double precision.\n\n");
    test_tridiagonall(&checksum);
    printf("\n\tQuadruple precision.\n\n");
    test_tridiagonalq(&checksum);
    printf("\n");
    return checksum;
}

void symmetric_tridiagonal_printmat(char * MAT, char * FMT, ft_symmetric_tridiagonal * A) {
    int n = A->n;
    double * a = A->a;
    double * b = A->b;

    printf("%s = \n", MAT);
    if (n == 1) {
        if (a[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, a[0]);
    }
    else if (n == 2) {
        if (a[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, a[0]);
        if (b[0] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[0]);
        printf("\n");
        if (b[0] < 0) {printf(" ");}
        else {printf("  ");}
        printf(FMT, b[0]);
        if (a[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, a[1]);
    }
    else if (n > 2) {
        // First row
        if (a[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, a[0]);
        if (b[0] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[0]);
        for (int j = 2; j < n; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }

        // Second row
        printf("\n");
        if (b[0] < 0) {printf(" ");}
        else {printf("  ");}
        printf(FMT, b[0]);
        if (a[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, a[1]);
        if (b[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[1]);
        for (int j = 3; j < n; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }

        // Interior rows
        for (int i = 2; i < n-1; i++) {
            printf("\n");
            printf("  ");
            printf(FMT, 0.0);
            for (int j = 1; j < i-1; j++) {
                printf("   ");
                printf(FMT, 0.0);
            }
            if (b[i-1] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, b[i-1]);
            if (a[i] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, a[i]);
            if (b[i] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, b[i]);
            for (int j = i+2; j < n; j++) {
                printf("   ");
                printf(FMT, 0.0);
            }
        }

        // Last row
        printf("\n");
        printf("  ");
        printf(FMT, 0.0);
        for (int j = 1; j < n-2; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }
        if (b[n-2] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[n-2]);
        if (a[n-1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, a[n-1]);
    }
    printf("]\n");
}


void bidiagonal_printmat(char * MAT, char * FMT, ft_bidiagonal * B) {
    int n = B->n;
    double * c = B->c;
    double * d = B->d;

    printf("%s = \n", MAT);
    if (n == 1) {
        if (c[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, c[0]);
    }
    else if (n == 2) {
        if (c[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, c[0]);
        if (d[0] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, d[0]);
        printf("\n");
        printf("  ");
        printf(FMT, 0.0);
        if (c[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, c[1]);
    }
    else if (n > 2) {
        // First row
        if (c[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, c[0]);
        if (d[0] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, d[0]);
        for (int j = 2; j < n; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }

        // Second row
        printf("\n");
        printf("  ");
        printf(FMT, 0.0);
        if (c[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, c[1]);
        if (d[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, d[1]);
        for (int j = 3; j < n; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }

        // Interior rows
        for (int i = 2; i < n-1; i++) {
            printf("\n");
            printf("  ");
            printf(FMT, 0.0);
            for (int j = 1; j < i; j++) {
                printf("   ");
                printf(FMT, 0.0);
            }
            if (c[i] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, c[i]);
            if (d[i] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, d[i]);
            for (int j = i+2; j < n; j++) {
                printf("   ");
                printf(FMT, 0.0);
            }
        }

        // Last row
        printf("\n");
        printf("  ");
        printf(FMT, 0.0);
        for (int j = 1; j < n-1; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }
        if (c[n-1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, c[n-1]);
    }
    printf("]\n");
}
