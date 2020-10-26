void X(checktest)(FLT err, FLT cst, int * checksum) {
    if (Y(fabs)(err) < cst*Y(eps)()) printf(GREEN("✓")"\n");
    else {printf(RED("✗")"\n"); (*checksum)++;}
}

FLT X(norm_1arg)(FLT * A, int n) {
    FLT ret = 0;
    for (int i = 0; i < n; i++)
        ret += A[i]*A[i];
    return Y(sqrt)(ret);
}

FLT X(norm_2arg)(FLT * A, FLT * B, int n) {
    FLT ret = 0;
    for (int i = 0; i < n; i++)
        ret += (A[i]-B[i])*(A[i]-B[i]);
    return Y(sqrt)(ret);
}

FLT X(normInf_1arg)(FLT * A, int n) {
    FLT ret = 0, temp;
    for (int i = 0; i < n; i++) {
        temp = Y(fabs)(A[i]);
        if (temp > ret) ret = temp;
    }
    return ret;
}

FLT X(normInf_2arg)(FLT * A, FLT * B, int n) {
    FLT ret = 0, temp;
    for (int i = 0; i < n; i++) {
        temp = Y(fabs)(A[i]-B[i]);
        if (temp > ret) ret = temp;
    }
    return ret;
}

FLT X(rec_A_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta) {
    if (norm) {
        if (n == 0)
            return (alpha+beta+2)/2*Y(sqrt)((alpha+beta+3)/((alpha+1)*(beta+1)));
        else
            return Y(sqrt)(((2*n+alpha+beta+1)*(2*n+alpha+beta+2)*(2*n+alpha+beta+2)*(2*n+alpha+beta+3))/((n+1)*(n+alpha+1)*(n+beta+1)*(n+alpha+beta+1)))/2;
    }
    else {
        if (n == 0)
            return (alpha+beta+2)/2;
        else
            return ((2*n+alpha+beta+1)*(2*n+alpha+beta+2))/(2*(n+1)*(n+alpha+beta+1));
    }
}

FLT X(rec_B_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta) {
    if (norm) {
        if (n == 0)
            return (alpha-beta)/2*Y(sqrt)((alpha+beta+3)/((alpha+1)*(beta+1)));
        else
            return ((alpha-beta)*(alpha+beta))/(2*(2*n+alpha+beta))*Y(sqrt)(((2*n+alpha+beta+1)*(2*n+alpha+beta+3))/((n+1)*(n+alpha+1)*(n+beta+1)*(n+alpha+beta+1)));
    }
    else {
        if (n == 0)
            return (alpha-beta)/2;
        else
            return ((alpha-beta)*(alpha+beta)*(2*n+alpha+beta+1))/(2*(n+1)*(n+alpha+beta+1)*(2*n+alpha+beta));
    }
}

FLT X(rec_C_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta) {
    if (norm) {
        if (n == 1)
            return (alpha+beta+4)/(alpha+beta+2)*Y(sqrt)(((alpha+1)*(beta+1)*(alpha+beta+5))/(2*(alpha+2)*(beta+2)*(alpha+beta+2)));
        else
            return (2*n+alpha+beta+2)/(2*n+alpha+beta)*Y(sqrt)((n*(n+alpha)*(n+beta)*(n+alpha+beta))/((n+1)*(n+alpha+1)*(n+beta+1)*(n+alpha+beta+1))*(2*n+alpha+beta+3)/(2*n+alpha+beta-1));
    }
    else {
        return ((n+alpha)*(n+beta)*(2*n+alpha+beta+2))/((n+1)*(n+alpha+beta+1)*(2*n+alpha+beta));
    }
}

FLT X(rec_A_laguerre)(const int norm, const int n, const FLT alpha) {
    if (norm) {
        return -1/Y(sqrt)((n+1)*(n+alpha+1));
    }
    else {
        return -ONE(FLT)/(n+1);
    }
}

FLT X(rec_B_laguerre)(const int norm, const int n, const FLT alpha) {
    if (norm) {
        return (2*n+alpha+1)/Y(sqrt)((n+1)*(n+alpha+1));
    }
    else {
        return (2*n+alpha+1)/(n+1);
    }
}

FLT X(rec_C_laguerre)(const int norm, const int n, const FLT alpha) {
    if (norm) {
        return Y(sqrt)((n*(n+alpha))/((n+1)*(n+alpha+1)));
    }
    else {
        return (n+alpha)/(n+1);
    }
}

FLT X(rec_A_hermite)(const int norm, const int n) {
    if (norm) {
        return Y(sqrt)(2/(n+ONE(FLT)));
    }
    else {
        return 2;
    }
}

FLT X(rec_B_hermite)(const int norm, const int n) {return 0;}

FLT X(rec_C_hermite)(const int norm, const int n) {
    if (norm) {
        return Y(sqrt)(n/(n+ONE(FLT)));
    }
    else {
        return 2*n;
    }
}
