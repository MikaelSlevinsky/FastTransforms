// Utility functions for testing.

#ifndef UTILITIESF_H
#define UTILITIESF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "fasttransformsf.h"

void printmat(char * MAT, float * A, int n, int m);
float vecnorm_1arg(float * A, int n, int m);
float vecnorm_2arg(float * A, float * B, int n, int m);
float vecnormInf_1arg(float * A, int n, int m);
float vecnormInf_2arg(float * A, float * B, int n, int m);
float rotnorm(const RotationPlan * RP);
float * sphones(int n, int m);
float * sphrand(int n, int m);
float * triones(int n, int m);
float * trirand(int n, int m);
float * diskones(int n, int m);
float * diskrand(int n, int m);
float * spinsphones(int n, int m, int s);
float * spinsphrand(int n, int m, int s);
float * copyA(float * A, int n, int m);
float elapsed(struct timeval * start, struct timeval * end, int N);

#endif //UTILITIESF_H
