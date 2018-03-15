// Computational routines for one-dimensional orthogonal polynomial transforms.

#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <stdlib.h>
#include <math.h>

#define M_SQRT_PI    1.772453850905516027   /* sqrt(pi)       */
#define M_1_SQRT_PI  0.564189583547756287   /* 1/sqrt(pi)     */
#define M_SQRT_PI_2  0.886226925452758014   /* sqrt(pi)/2     */

static inline double stirlingseries(const double z);
static inline double lambda(const double x);
static inline double lambda2(const double x, const double l1, const double l2);

double * plan_leg2cheb(const int normleg, const int normcheb, const int n);
double * plan_cheb2leg(const int normcheb, const int normleg, const int n);
double * plan_ultra2ultra(const int normultra1, const int normultra2, const int n, const double lambda1, const double lambda2);

#endif //TRANSFORMS_H
