// Utility functions for testing.

#ifndef FTUTILITIES_H
#define FTUTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "ftinternal.h"

#define FT_CONCAT(prefix, name, suffix) prefix ## name ## suffix

#define FT_TIME(BLOCK, START, END, NTIMES)                                     \
BLOCK;                                                                         \
BLOCK;                                                                         \
gettimeofday(&START, NULL);                                                    \
for (int ntimes = 0; ntimes < NTIMES; ntimes++)                                \
    BLOCK;                                                                     \
gettimeofday(&END, NULL);

// A bitwise OR ('|') of zero or more of the following: FFTW_ESTIMATE FFTW_MEASURE FFTW_PATIENT FFTW_EXHAUSTIVE FFTW_WISDOM_ONLY FFTW_DESTROY_INPUT FFTW_PRESERVE_INPUT FFTW_UNALIGNED
#define FT_FFTW_FLAGS FFTW_MEASURE | FFTW_DESTROY_INPUT

void printmat(char * MAT, char * FMT, double * A, int n, int m);
void print_summary_size(size_t i);
double * copymat(double * A, int n, int m);
double * sphones(int n, int m);
double * sphrand(int n, int m);
double * triones(int n, int m);
double * trirand(int n, int m);
double * diskones(int n, int m);
double * diskrand(int n, int m);
double * rectdiskones(int n, int m);
double * rectdiskrand(int n, int m);
double * tetones(int n, int l, int m);
double * tetrand(int n, int l, int m);
ft_complex * spinsphones(int n, int m, int s);
ft_complex * spinsphrand(int n, int m, int s);
double elapsed(struct timeval * start, struct timeval * end, int N);

#define FLT float
#define X(name) FT_CONCAT(ft_, name, f)
#define Y(name) FT_CONCAT(, name, f)
#include "ftutilities_source.h"
#undef FLT
#undef X
#undef Y

#define FLT double
#define X(name) FT_CONCAT(ft_, name, )
#define Y(name) FT_CONCAT(, name, )
#include "ftutilities_source.h"
#undef FLT
#undef X
#undef Y

#define FLT long double
#define X(name) FT_CONCAT(ft_, name, l)
#define Y(name) FT_CONCAT(, name, l)
#include "ftutilities_source.h"
#undef FLT
#undef X
#undef Y

#if defined(FT_QUADMATH)
    #define FLT quadruple
    #define X(name) FT_CONCAT(ft_, name, q)
    #define Y(name) FT_CONCAT(, name, q)
    #include "ftutilities_source.h"
    #undef FLT
    #undef X
    #undef Y
#endif

#endif //FTUTILITIES_H
