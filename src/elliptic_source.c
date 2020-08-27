static inline FLT X(arithmetic_geometric_mean)(FLT a, FLT b) {
    FLT c = Y(sqrt)(Y(fabs)((a-b)*(a+b)));
    while (c > 2*MAX(Y(fabs)(a), Y(fabs)(b))*Y(eps)()) {
        FLT t = (a+b)/2;
        b = Y(sqrt)(a*b);
        a = t;
        c *= c/(4*a);
    }
    return a;
}

static inline FLT X(arithmetic_geometric_mean_with_accumulator)(FLT a, FLT b, FLT * s) {
    FLT c = Y(sqrt)(Y(fabs)((a-b)*(a+b)));
    * s = a*a-c*c/2;
    FLT two = 2;
    FLT pow2 = 1/two;
    while (c > 2*MAX(Y(fabs)(a), Y(fabs)(b))*Y(eps)()) {
        FLT t = (a+b)/2;
        b = Y(sqrt)(a*b);
        a = t;
        c *= c/(4*a);
        pow2 *= two;
        * s -= pow2*c*c;
    }
    return a;
}

FLT X(complete_elliptic_integral)(char KIND, FLT k) {
    FLT kp = Y(sqrt)((1-k)*(1+k));
    if (KIND == '1')
        if (Y(fabs)(kp) < 2*Y(fabs)(k)*Y(eps)())
            return ONE(FLT)/ZERO(FLT);
        else
            return 2*Y(atan)(1)/X(arithmetic_geometric_mean)(ONE(FLT), kp);
    else if (KIND == '2')
        if (Y(fabs)(kp) < 2*Y(fabs)(k)*Y(eps)())
            return ONE(FLT);
        else {
            FLT s;
            FLT x = X(arithmetic_geometric_mean_with_accumulator)(ONE(FLT), kp, &s);
            return 2*Y(atan)(1)/x*s;
        }
    else return ONE(FLT)/ZERO(FLT);
}

void X(jacobian_elliptic_functions)(const FLT x, const FLT k, FLT * sn, FLT * cn, FLT * dn, unsigned flags) {
    FLT kp = Y(sqrt)((1-k)*(1+k));
    FLT a = 1, b = kp, c = k;
    int n = 0;
    while (c > 2*Y(eps)()) {
        FLT t = (a+b)/2;
        b = Y(sqrt)(a*b);
        a = t;
        c *= c/(4*a);
        n++;
    }
    FLT phi = Y(pow)(2, n)*a*x;
    for (int i = n-1; i >= 0; i--) {
        phi = (phi + Y(asin)(c/a*Y(sin)(phi)))/2;
        FLT t = a+c;
        c = 2*Y(sqrt)(a*c);
        a = t;
    }
    if (flags & FT_SN)
        * sn = Y(sin)(phi);
    if (flags & FT_CN)
        * cn = Y(cos)(phi);
    if (flags & FT_DN)
        * dn = Y(sqrt)((1-k*Y(sin)(phi))*(1+k*Y(sin)(phi)));
}
