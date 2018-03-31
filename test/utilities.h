// Utility functions for testing.

#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include "../src/rotations.h"

void printmat(char * MAT, double * A, int n, int m);
double vecnorm_1arg(double * A, int n, int m);
double vecnorm_2arg(double * A, double * B, int n, int m);
double vecnormInf_1arg(double * A, int n, int m);
double vecnormInf_2arg(double * A, double * B, int n, int m);
double rotnorm(const RotationPlan * RP);
double * sphones(int n, int m);
double * sphrand(int n, int m);
double * triones(int n, int m);
double * trirand(int n, int m);
double * copyA(double * A, int n, int m);

#endif //UTILITIES_H
