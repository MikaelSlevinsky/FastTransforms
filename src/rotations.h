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

void kernel1_sph_hi2lo(const RotationPlan * RP, const int m, double * A);
void kernel1_sph_lo2hi(const RotationPlan * RP, const int m, double * A);

void kernel2_sph_hi2lo(const RotationPlan * RP, const int m, double * A, double * B);
void kernel2_sph_lo2hi(const RotationPlan * RP, const int m, double * A, double * B);

void kernel2x4_sph_hi2lo(const RotationPlan * RP, const int m, double * A, double * B);
void kernel2x4_sph_lo2hi(const RotationPlan * RP, const int m, double * A, double * B);

static inline void apply_givens_1x1(const double * s, const double * c, const int n, const int l, const int m, double * A);
static inline void apply_givens_t_1x1(const double * s, const double * c, const int n, const int l, const int m, double * A);

static inline void apply_givens_2x1(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);
static inline void apply_givens_t_2x1(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);

static inline void apply_givens_2x2(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);
static inline void apply_givens_t_2x2(const double * s, const double * c, const int n, const int l, const int m, double * A, double * B);

#endif //ROTATIONS_H
