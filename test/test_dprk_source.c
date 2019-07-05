static FLT X(diff)(FLT x, FLT y) {return x - y;}

FLT X(secular_cond)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, ret = ZERO(FLT);
    for (int i = 0; i < n; i++)
        ret += X(fabs)(z[i]*z[i]/(X(diff)(d[i], lambdahi) - lambdalo));
    return X(fabs)(1/rho)+ret;
}

FLT X(generalized_secular_cond)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma, ret = ZERO(FLT);
    for (int i = 0; i < n; i++)
        ret += X(fabs)(z[i]*z[i]/(X(diff)(d[i], lambdahi) - lambdalo));
    return X(fabs)(1/(sigma*(X(diff)(rho/sigma, lambdahi) - lambdalo)))+ret;
}

void X(test_permute)(int * checksum) {
    int n = 10;
    int p[] = {1,2,4,6,8,0,3,5,7,9};
    FLT x[] = {0.123,0.456,0.789,0.135,0.246,0.791,0.802,0.147,0.258,0.369}, y[10], z[10];
    for (int i = 0; i < n; i++)
        y[i] = x[p[i]];
    for (int i = 0; i < n; i++)
        z[p[i]] = y[i];
    FLT err = X(norm_2arg)(x, z, n)/X(norm_1arg)(x, n);
    printf("Comparison of direct row permutations \t\t\t |%20i ", (int) err);
    X(checktest)(err, n, checksum);
    X(perm)('N', x, p, n);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(x, n);
    X(perm)('T', x, p, n);
    err += X(norm_2arg)(x, z, n)/X(norm_1arg)(x, n);
    printf("Comparison of in-place row permutations \t\t |%20i ", (int) err);
    X(checktest)(err, n, checksum);
}

X(symmetric_dpr1) * X(test_symmetric_dpr1)(int n) {
    X(symmetric_dpr1) * A = (X(symmetric_dpr1) *) malloc(sizeof(X(symmetric_dpr1)));
    A->d = (FLT *) calloc(n, sizeof(FLT));
    A->z = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++) {
        A->d[i] = (i+1)*(i+2);
        A->z[i] = i+1;
    }
    A->rho = 1.0;
    A->n = n;
    return A;
}

X(symmetric_idpr1) * X(test_symmetric_idpr1)(int n) {
    X(symmetric_idpr1) * B = (X(symmetric_idpr1) *) malloc(sizeof(X(symmetric_idpr1)));
    B->z = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        B->z[i] = i+1;
    B->sigma = 1.0;
    B->n = n;
    return B;
}


void X(test_dprk)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 256;
    struct timeval start, end;

    X(test_permute)(checksum);

    FLT * Id, * AC, * Q, * QtQ, * QtAQ, * V, * VtBV, * VtAV, * flam;

    X(symmetric_dpr1) * A, * Ac, * C, * Cc;
    X(symmetric_idpr1) * B, * Bc;

    X(symmetric_dpr1_eigen) * F;
    X(symmetric_dpr1_eigen_FMM) * HF;

    A = X(test_symmetric_dpr1)(n);
    B = X(test_symmetric_idpr1)(n);
    C = X(symmetric_dpr1_inv)(A);

    Id = (FLT *) calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        Id[j+j*n] = 1;

    AC = (FLT *) calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        AC[j+j*n] = 1;
        X(drmv)('N', A, AC+j*n);
        X(drmv)('N', C, AC+j*n);
    }
    FLT err = X(norm_2arg)(AC, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in ||A⁻¹A - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_dpr1)(A);
    X(destroy_symmetric_dpr1)(C);


    A = X(test_symmetric_dpr1)(n);
    Ac = X(test_symmetric_dpr1)(n);
    F = X(symmetric_dpr1_eig)(A);

    Q = (FLT *) calloc(n*n, sizeof(FLT));
    QtQ = (FLT *) calloc(n*n, sizeof(FLT));
    QtAQ = (FLT *) calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        X(dvmv)('N', 1, F, Id+j*n, 0, Q+j*n);
    for (int j = 0; j < n; j++)
        X(dvmv)('T', 1, F, Q+j*n, 0, QtQ+j*n);
    for (int j = 0; j < n; j++)
        X(drmv)('N', Ac, Q+j*n);
    for (int j = 0; j < n; j++)
        X(dvmv)('T', 1, F, Q+j*n, 0, QtAQ+j*n);
    for (int j = 0; j < n; j++)
        QtAQ[j+j*n] -= F->lambda[j];

    flam = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        flam[i] = X(secular)(Ac, F->lambdalo[i], F->lambdahi[i])/X(secular_cond)(Ac, F->lambdalo[i], F->lambdahi[i]);
    err = X(norm_1arg)(flam, n);
    printf("Eigenvalue error w.r.t. condition number f(λ)/κ_f(λ) \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_2arg)(QtQ, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in ||QᵀQ - I|| / ||I|| \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(QtAQ, n*n)/X(norm_1arg)(F->lambda, n);
    printf("Numerical error in ||QᵀAQ - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_dpr1)(A);
    X(destroy_symmetric_dpr1)(Ac);
    X(destroy_symmetric_dpr1_eigen)(F);


    A = X(test_symmetric_dpr1)(n);
    Ac = X(test_symmetric_dpr1)(n);
    HF = X(symmetric_dpr1_eig_FMM)(A);

    for (int j = 0; j < n; j++)
        X(dfmv)('N', 1, HF, Id+j*n, 0, Q+j*n);
    for (int j = 0; j < n; j++)
        X(dfmv)('T', 1, HF, Q+j*n, 0, QtQ+j*n);
    for (int j = 0; j < n; j++)
        X(drmv)('N', Ac, Q+j*n);
    for (int j = 0; j < n; j++)
        X(dfmv)('T', 1, HF, Q+j*n, 0, QtAQ+j*n);
    for (int j = 0; j < n; j++)
        QtAQ[j+j*n] -= HF->lambda[j];

    for (int i = 0; i < n; i++)
        flam[i] = X(secular)(Ac, HF->lambdalo[i], HF->lambdahi[i])/X(secular_cond)(Ac, HF->lambdalo[i], HF->lambdahi[i]);
    err = X(norm_1arg)(flam, n);
    printf("Eigenvalue FMM'ed error f(λ)/κ_f(λ) \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_2arg)(QtQ, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in FMM'ed ||QᵀQ - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(QtAQ, n*n)/X(norm_1arg)(HF->lambda, n);
    printf("Numerical error in FMM'ed ||QᵀAQ - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_dpr1)(A);
    X(destroy_symmetric_dpr1)(Ac);
    X(destroy_symmetric_dpr1_eigen_FMM)(HF);

    A = X(test_symmetric_dpr1)(n);
    Ac = X(test_symmetric_dpr1)(n);
    B = X(test_symmetric_idpr1)(n);
    Bc = X(test_symmetric_idpr1)(n);

    F = X(symmetric_definite_dpr1_eig)(A, B);

    V = (FLT *) calloc(n*n, sizeof(FLT));
    VtBV = (FLT *) calloc(n*n, sizeof(FLT));
    VtAV = (FLT *) calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        X(dvmv)('N', 1, F, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(irmv)('N', Bc, V+j*n);
    for (int j = 0; j < n; j++)
        X(dvmv)('T', 1, F, V+j*n, 0, VtBV+j*n);
    for (int j = 0; j < n; j++)
        X(dvmv)('N', 1, F, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(drmv)('N', Ac, V+j*n);
    for (int j = 0; j < n; j++)
        X(dvmv)('T', 1, F, V+j*n, 0, VtAV+j*n);
    for (int j = 0; j < n; j++)
        VtAV[j+j*n] -= F->lambda[j];

    for (int i = 0; i < n; i++)
        flam[i] = X(generalized_secular)(Ac, Bc, F->lambdalo[i], F->lambdahi[i])/X(generalized_secular_cond)(Ac, Bc, F->lambdalo[i], F->lambdahi[i]);
    err = X(norm_1arg)(flam, n);
    printf("Generalized eigenvalue error f(λ)/κ_f(λ) \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_2arg)(VtBV, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in ||VᵀBV - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(F->lambda, n);
    printf("Numerical error in ||VᵀAV - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_dpr1)(A);
    X(destroy_symmetric_dpr1)(Ac);
    X(destroy_symmetric_idpr1)(B);
    X(destroy_symmetric_idpr1)(Bc);
    X(destroy_symmetric_dpr1_eigen)(F);

    A = X(test_symmetric_dpr1)(n);
    Ac = X(test_symmetric_dpr1)(n);
    B = X(test_symmetric_idpr1)(n);
    Bc = X(test_symmetric_idpr1)(n);

    HF = X(symmetric_definite_dpr1_eig_FMM)(A, B);

    for (int j = 0; j < n; j++)
        X(dfmv)('N', 1, HF, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(irmv)('N', Bc, V+j*n);
    for (int j = 0; j < n; j++)
        X(dfmv)('T', 1, HF, V+j*n, 0, VtBV+j*n);
    for (int j = 0; j < n; j++)
        X(dfmv)('N', 1, HF, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(drmv)('N', Ac, V+j*n);
    for (int j = 0; j < n; j++)
        X(dfmv)('T', 1, HF, V+j*n, 0, VtAV+j*n);
    for (int j = 0; j < n; j++)
        VtAV[j+j*n] -= HF->lambda[j];

    for (int i = 0; i < n; i++)
        flam[i] = X(generalized_secular)(Ac, Bc, HF->lambdalo[i], HF->lambdahi[i])/X(generalized_secular_cond)(Ac, Bc, HF->lambdalo[i], HF->lambdahi[i]);
    err = X(norm_1arg)(flam, n);
    printf("Generalized eigenvalue FMM'ed error f(λ)/κ_f(λ) \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_2arg)(VtBV, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in FMM'ed ||VᵀBV - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(HF->lambda, n);
    printf("Numerical error in FMM'ed ||VᵀAV - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_dpr1)(A);
    X(destroy_symmetric_dpr1)(Ac);
    X(destroy_symmetric_idpr1)(B);
    X(destroy_symmetric_idpr1)(Bc);
    X(destroy_symmetric_dpr1_eigen_FMM)(HF);

    A = X(test_symmetric_dpr1)(n);
    C = X(symmetric_dpr1_inv)(A);
    Cc = X(symmetric_dpr1_inv)(A);
    F = X(symmetric_dpr1_eig)(C);

    for (int i = 0; i < n; i++)
        flam[i] = X(secular)(Cc, F->lambdalo[i], F->lambdahi[i])/X(secular_cond)(Cc, F->lambdalo[i], F->lambdahi[i]);
    err = X(norm_1arg)(flam, n);
    printf("Inverse eigenvalue error f(λ)/κ_f(λ) \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_dpr1)(A);
    X(destroy_symmetric_dpr1)(C);
    X(destroy_symmetric_dpr1)(Cc);
    X(destroy_symmetric_dpr1_eigen)(F);

    free(Id);
    free(AC);
    free(Q);
    free(QtQ);
    free(QtAQ);
    free(V);
    free(VtBV);
    free(VtAV);
    free(flam);
}
