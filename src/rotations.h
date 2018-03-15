// Computational routines for the harmonic polynomial connection problem.

#ifndef ROTATIONS_H
#define ROTATIONS_H

#include <stdlib.h>
#include <math.h>

typedef struct {
    double * s;
    double * c;
    int n;
} RotationPlan;

RotationPlan * plan_rotsphere(const int n);
RotationPlan * plan_rotspinsphere(const int n, const int m1, const int m2);
RotationPlan * plan_rotdisk(const int n);
RotationPlan * plan_rottriangle(const int n, const double alpha, const double beta, const double gamma);

void kernel1_sph_hi2lo(const RotationPlan * RP, double * A, const int m);
void kernel1_sph_lo2hi(const RotationPlan * RP, double * A, const int m);

void kernel2_sph_hi2lo(const RotationPlan * RP, double * A, double * B, const int m);
void kernel2_sph_lo2hi(const RotationPlan * RP, double * A, double * B, const int m);

#endif //ROTATIONS_H
