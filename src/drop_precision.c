X(tdc_eigen) * X(drop_precision_tdc_eigen)(X2(tdc_eigen) * F2) {
    int n = F2->n;
    X(tdc_eigen) * F = malloc(sizeof(X(tdc_eigen)));
    if (n < TDC_EIGEN_BLOCKSIZE) {
        FLT * V = malloc(n*n*sizeof(FLT));
        for (int i = 0; i < n*n; i++)
            V[i] = F2->V[i];
        FLT * lambda = malloc(n*sizeof(FLT));
        for (int i = 0; i < n; i++)
            lambda[i] = F2->lambda[i];
        F->V = V;
        F->lambda = lambda;
        F->n = n;
    }
    else {
        F->F0 = X(drop_precision_symmetric_dpr1_eigen)(F2->F0);
        F->F1 = X(drop_precision_tdc_eigen)(F2->F1);
        F->F2 = X(drop_precision_tdc_eigen)(F2->F2);
        F->z = calloc(n, sizeof(FLT));
        F->n = n;
    }
    return F;
}

X(tdc_eigen_FMM) * X(drop_precision_tdc_eigen_FMM)(X2(tdc_eigen_FMM) * F2) {
    int n = F2->n;
    X(tdc_eigen_FMM) * F = malloc(sizeof(X(tdc_eigen_FMM)));
    if (n < TDC_EIGEN_BLOCKSIZE) {
        FLT * V = malloc(n*n*sizeof(FLT));
        for (int i = 0; i < n*n; i++)
            V[i] = F2->V[i];
        FLT * lambda = malloc(n*sizeof(FLT));
        for (int i = 0; i < n; i++)
            lambda[i] = F2->lambda[i];
        F->V = V;
        F->lambda = lambda;
        F->n = n;
    }
    else {
        F->F0 = X(drop_precision_symmetric_dpr1_eigen_FMM)(F2->F0);
        F->F1 = X(drop_precision_tdc_eigen_FMM)(F2->F1);
        F->F2 = X(drop_precision_tdc_eigen_FMM)(F2->F2);
        F->z = calloc(n, sizeof(FLT));
        F->n = n;
    }
    return F;
}

X(symmetric_dpr1_eigen) * X(drop_precision_symmetric_dpr1_eigen)(X2(symmetric_dpr1_eigen) * F2) {
    int n = F2->n, iz = F2->iz, id = F2->id;
    int * p = malloc(n*sizeof(int)), * q = malloc(n*sizeof(int));
    FLT * lambda = malloc(n*sizeof(FLT)), * lambdalo = malloc(n*sizeof(FLT)), * lambdahi = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++) {
        p[i] = F2->p[i];
        q[i] = F2->q[i];
        lambda[i] = F2->lambda[i];
        lambdalo[i] = F2->lambdalo[i];
        lambdahi[i] = F2->lambdahi[i];
    }
    FLT * v = malloc(id*sizeof(FLT));
    for (int i = 0; i < id; i++)
        v[i] = F2->v[i];
    FLT * V = malloc((n-iz)*(n-iz-id)*sizeof(FLT));
    for (int i = 0; i < (n-iz)*(n-iz-id); i++)
        V[i] = F2->V[i];
    X(symmetric_dpr1_eigen) * F = malloc(sizeof(X(symmetric_dpr1_eigen)));
    F->v = v;
    F->V = V;
    F->lambda = lambda;
    F->lambdalo = lambdalo;
    F->lambdahi = lambdahi;
    F->p = p;
    F->q = q;
    F->n = n;
    F->iz = iz;
    F->id = id;
    return F;
}

X(symmetric_dpr1_eigen_FMM) * X(drop_precision_symmetric_dpr1_eigen_FMM)(X2(symmetric_dpr1_eigen_FMM) * F2) {
    int n = F2->n, iz = F2->iz, id = F2->id;
    int * p = malloc(n*sizeof(int)), * q = malloc(n*sizeof(int));
    FLT * lambda = malloc(n*sizeof(FLT)), * lambdalo = malloc(n*sizeof(FLT)), * lambdahi = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++) {
        p[i] = F2->p[i];
        q[i] = F2->q[i];
        lambda[i] = F2->lambda[i];
        lambdalo[i] = F2->lambdalo[i];
        lambdahi[i] = F2->lambdahi[i];
    }
    FLT * v = malloc(id*sizeof(FLT));
    for (int i = 0; i < id; i++)
        v[i] = F2->v[i];

    X(symmetric_dpr1) * A = malloc(sizeof(X(symmetric_dpr1)));
    X(symmetric_idpr1) * B = malloc(sizeof(X(symmetric_idpr1)));
    B->n = A->n = F2->A->n;
    A->d = malloc(A->n*sizeof(FLT));
    A->z = malloc(A->n*sizeof(FLT));
    B->z = malloc(A->n*sizeof(FLT));
    for (int i = 0; i < A->n; i++) {
        A->d[i] = F2->A->d[i];
        B->z[i] = A->z[i] = F2->A->z[i];
    }
    A->rho = F2->A->rho;
    B->sigma = F2->B->sigma;

    X(perm)('T', lambda, q, n);
    X(perm)('T', lambdalo, q, n);
    X(perm)('T', lambdahi, q, n);
    X(hierarchicalmatrix) * V = X(symmetric_definite_dpr1_eigvecs_FMM)(A, B, lambda+iz+id, lambdalo+iz+id, lambdahi+iz+id, n-iz-id);
    X(perm)('N', lambda, q, n);
    X(perm)('N', lambdalo, q, n);
    X(perm)('N', lambdahi, q, n);

    X(symmetric_dpr1_eigen_FMM) * F = malloc(sizeof(X(symmetric_dpr1_eigen_FMM)));
    F->A = A;
    F->B = B;
    F->v = v;
    F->V = V;
    F->lambda = lambda;
    F->lambdalo = lambdalo;
    F->lambdahi = lambdahi;
    F->p = p;
    F->q = q;
    F->n = n;
    F->iz = iz;
    F->id = id;
    return F;
}

X(tb_eigen_FMM) * X(drop_precision_tb_eigen_FMM)(X2(tb_eigen_FMM) * F2) {
    int n = F2->n;
    X(tb_eigen_FMM) * F = malloc(sizeof(X(tb_eigen_FMM)));
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT * V = malloc(n*n*sizeof(FLT));
        for (int i = 0; i < n*n; i++)
            V[i] = F2->V[i];
        FLT * lambda = malloc(n*sizeof(FLT));
        for (int i = 0; i < n; i++)
            lambda[i] = F2->lambda[i];
        F->V = V;
        F->lambda = lambda;
        F->n = n;
    }
    else {
        int s = n>>1, b = F2->b;
        FLT * lambda = malloc(n*sizeof(FLT));
        for (int i = 0; i < n; i++)
            lambda[i] = F2->lambda[i];
        F->F0 = X(sample_hierarchicalmatrix)(X(cauchykernel), lambda, lambda+s, (unitrange) {0, s}, (unitrange) {0, n-s});
        F->F1 = X(drop_precision_tb_eigen_FMM)(F2->F1);
        F->F2 = X(drop_precision_tb_eigen_FMM)(F2->F2);
        F->X = malloc(s*b*sizeof(FLT));
        for (int i = 0; i < s*b; i++)
            F->X[i] = F2->X[i];
        F->Y = malloc((n-s)*b*sizeof(FLT));
        for (int i = 0; i < (n-s)*b; i++)
            F->Y[i] = F2->Y[i];
        F->t1 = calloc(s*FT_GET_MAX_THREADS(), sizeof(FLT));
        F->t2 = calloc((n-s)*FT_GET_MAX_THREADS(), sizeof(FLT));
        F->lambda = lambda;
        F->n = n;
        F->b = b;
    }
    return F;
}
