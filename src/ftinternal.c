#include "ftinternal.h"

float epsf(void) {return (float) M_EPSf;}
double eps(void) {return (double) M_EPS;}
long double epsl(void) {return (long double) M_EPSl;}
quadruple epsq(void) {return (quadruple) M_EPSq;}

#if !(__APPLE__)
    float __cospif(float x) {return cosf(M_PIf*x);}
    double __cospi(double x) {return cos(M_PI*x);}
    float __sinpif(float x) {return sinf(M_PIf*x);}
    double __sinpi(double x) {return sin(M_PI*x);}
    float __tanpif(float x) {return tanf(M_PIf*x);}
    double __tanpi(double x) {return tan(M_PI*x);}
#endif
long double __cospil(long double x) {return cosl(M_PIl*x);}
quadruple __cospiq(quadruple x) {return cosq(M_PIq*x);}
long double __sinpil(long double x) {return sinl(M_PIl*x);}
quadruple __sinpiq(quadruple x) {return sinq(M_PIq*x);}
long double __tanpil(long double x) {return tanl(M_PIl*x);}
quadruple __tanpiq(quadruple x) {return tanq(M_PIq*x);}
