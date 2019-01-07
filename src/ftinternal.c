#include "ftinternal.h"

float epsf(void) {return (float) M_EPSf;}
double eps(void) {return (double) M_EPS;}
long double epsl(void) {return (long double) M_EPSl;}
quadruple epsq(void) {return (quadruple) M_EPSq;}

long double __cospil(long double x) {return cosl(M_PIl*x);}
quadruple __cospiq(quadruple x) {return cosq(M_PIq*x);}
long double __sinpil(long double x) {return sinl(M_PIl*x);}
quadruple __sinpiq(quadruple x) {return sinq(M_PIq*x);}
long double __tanpil(long double x) {return tanl(M_PIl*x);}
quadruple __tanpiq(quadruple x) {return tanq(M_PIq*x);}
