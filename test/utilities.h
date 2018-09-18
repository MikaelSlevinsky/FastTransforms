// Utility functions for testing.

#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "fasttransforms.h"
#include "ftinternal.h"

void printmat(char * MAT, double * A, int n, int m);
double vecnorm_1arg(double * A, int n, int m);
double vecnorm_2arg(double * A, double * B, int n, int m);
double vecnormInf_1arg(double * A, int n, int m);
double vecnormInf_2arg(double * A, double * B, int n, int m);
double rotnorm(const ft_rotation_plan * RP);
double * sphones(int n, int m);
double * sphrand(int n, int m);
double * triones(int n, int m);
double * trirand(int n, int m);
double * diskones(int n, int m);
double * diskrand(int n, int m);
double * spinsphones(int n, int m, int s);
double * spinsphrand(int n, int m, int s);
double * copyA(double * A, int n, int m);
double * copyAlign(double * A, int n, int m);
double elapsed(struct timeval * start, struct timeval * end, int N);

#endif //UTILITIES_H
