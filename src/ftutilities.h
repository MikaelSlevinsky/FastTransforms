// Utility functions for testing.

#ifndef FTUTILITIES_H
#define FTUTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define RESET   "\x1b[0m"

void printmat(char * MAT, char * FMT, double * A, int n, int m);
double * copymat(double * A, int n, int m);
double vecnorm_1arg(double * A, int n, int m);
double vecnorm_2arg(double * A, double * B, int n, int m);
double vecnormInf_1arg(double * A, int n, int m);
double vecnormInf_2arg(double * A, double * B, int n, int m);
double * sphones(int n, int m);
double * sphrand(int n, int m);
double * triones(int n, int m);
double * trirand(int n, int m);
double * diskones(int n, int m);
double * diskrand(int n, int m);
double * spinsphones(int n, int m, int s);
double * spinsphrand(int n, int m, int s);
double elapsed(struct timeval * start, struct timeval * end, int N);

#endif //FTUTILITIES_H
