void X(checktest)(FLT err, int n, int * checksum) {
    if (X(fabs)(err) < n*X(eps)()) printf(GREEN("✓")"\n");
    else {printf(RED("×")"\n"); (*checksum)++;}
}

FLT X(norm_1arg)(FLT * A, int n) {
    FLT ret = 0;
    for (int i = 0; i < n; i++)
        ret += X(pow)(A[i], 2);
    return X(sqrt)(ret);
}

FLT X(norm_2arg)(FLT * A, FLT * B, int n) {
    FLT ret = 0;
    for (int i = 0; i < n; i++)
        ret += X(pow)(A[i]-B[i], 2);
    return X(sqrt)(ret);
}

FLT X(normInf_1arg)(FLT * A, int n) {
    FLT ret = 0, temp;
    for (int i = 0; i < n; i++) {
        temp = X(fabs)(A[i]);
        if (temp > ret) ret = temp;
    }
    return ret;
}

FLT X(normInf_2arg)(FLT * A, FLT * B, int n) {
    FLT ret = 0, temp;
    for (int i = 0; i < n; i++) {
        temp = X(fabs)(A[i]-B[i]);
        if (temp > ret) ret = temp;
    }
    return ret;
}
