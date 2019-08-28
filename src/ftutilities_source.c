void X(checktest)(FLT err, FLT cst, int * checksum) {
    if (Y(fabs)(err) < cst*Y(eps)()) printf(GREEN("✓")"\n");
    else {printf(RED("×")"\n"); (*checksum)++;}
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
