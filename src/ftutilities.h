// Utility functions for testing.

#ifndef FTUTILITIES_H
#define FTUTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define RED(string) "\x1b[31m" string "\x1b[0m"
#define GREEN(string) "\x1b[32m" string "\x1b[0m"
#define YELLOW(string) "\x1b[33m" string "\x1b[0m"
#define BLUE(string) "\x1b[34m" string "\x1b[0m"
#define MAGENTA(string) "\x1b[35m" string "\x1b[0m"
#define CYAN(string) "\x1b[36m" string "\x1b[0m"

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
