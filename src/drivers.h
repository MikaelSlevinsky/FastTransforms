// Driver routines for synthesis and analysis of harmonic polynomial transforms.

#ifndef DRIVERS_H
#define DRIVERS_H

#include <cblas.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
#endif

#include "rotations.h"
#include "transforms.h"

typedef struct {
    RotationPlan * RP;
    double * P1;
    double * P2;
    double * P1inv;
    double * P2inv;
} SphericalHarmonicPlan;

void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M);

SphericalHarmonicPlan * plan_sph2fourier(const int n);

void execute_sph2fourier(const SphericalHarmonicPlan * P, double * A, const int N, const int M);
void execute_fourier2sph(const SphericalHarmonicPlan * P, double * A, const int N, const int M);

#endif //DRIVERS_H
