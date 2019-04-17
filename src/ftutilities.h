// Utility functions for testing.

#ifndef FTUTILITIES_H
#define FTUTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "ftinternal.h"

#define RED(string) "\x1b[31m" string "\x1b[0m"
#define GREEN(string) "\x1b[32m" string "\x1b[0m"
#define YELLOW(string) "\x1b[33m" string "\x1b[0m"
#define BLUE(string) "\x1b[34m" string "\x1b[0m"
#define MAGENTA(string) "\x1b[35m" string "\x1b[0m"
#define CYAN(string) "\x1b[36m" string "\x1b[0m"

void printmat(char * MAT, char * FMT, double * A, int n, int m);
double * copymat(double * A, int n, int m);
double * sphones(int n, int m);
double * sphrand(int n, int m);
double * triones(int n, int m);
double * trirand(int n, int m);
double * diskones(int n, int m);
double * diskrand(int n, int m);
double * tetones(int n, int l, int m);
double * tetrand(int n, int l, int m);
double * spinsphones(int n, int m, int s);
double * spinsphrand(int n, int m, int s);
double elapsed(struct timeval * start, struct timeval * end, int N);

#define FLT float
#define X(name) CONCAT(, name, f)
#include "ftutilities_source.h"
#undef FLT
#undef X

#define FLT double
#define X(name) CONCAT(, name, )
#include "ftutilities_source.h"
#undef FLT
#undef X

#define FLT long double
#define X(name) CONCAT(, name, l)
#include "ftutilities_source.h"
#undef FLT
#undef X

#define FLT quadruple
#define X(name) CONCAT(, name, q)
#include "ftutilities_source.h"
#undef FLT
#undef X

#endif //FTUTILITIES_H
