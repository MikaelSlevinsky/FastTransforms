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

void execute_sph_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_sph_lo2hi(const RotationPlan * RP, double * A, const int M);

void execute_tri_hi2lo(const RotationPlan * RP, double * A, const int M);
void execute_tri_lo2hi(const RotationPlan * RP, double * A, const int M);

typedef struct {
    RotationPlan * RP;
    double * P1;
    double * P2;
    double * P1inv;
    double * P2inv;
} SphericalHarmonicPlan;

SphericalHarmonicPlan * plan_sph2fourier(const int n);

void execute_sph2fourier(const SphericalHarmonicPlan * P, double * A, const int N, const int M);
void execute_fourier2sph(const SphericalHarmonicPlan * P, double * A, const int N, const int M);

typedef struct {
    RotationPlan * RP;
    double * P1;
    double * P2;
    double * P3;
    double * P4;
    double * P1inv;
    double * P2inv;
    double * P3inv;
    double * P4inv;
    double alpha;
    double beta;
    double gamma;
} TriangularHarmonicPlan;

TriangularHarmonicPlan * plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma);

void execute_tri2cheb(const TriangularHarmonicPlan * P, double * A, const int N, const int M);
void execute_cheb2tri(const TriangularHarmonicPlan * P, double * A, const int N, const int M);

void alternate_sign(double * A, const int N);

void chebyshev_normalization(double * A, const int N, const int M);
void chebyshev_normalization_t(double * A, const int N, const int M);

#endif //DRIVERS_H
