void X(test_tridiagonal)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 50;

    X(symmetric_tridiagonal) * A = (X(symmetric_tridiagonal) *) malloc(sizeof(X(symmetric_tridiagonal)));
    FLT * a = (FLT *) calloc(n, sizeof(FLT));
    FLT * b = (FLT *) calloc(n-1, sizeof(FLT));
    for (int i = 0; i < n; i++)
        a[i] = 2;
    for (int i = 0; i < n-1; i++)
        b[i] = -1;
    A->a = a;
    A->b = b;
    A->n = n;

    FLT alpha = 2;
    FLT beta = 0.5;
    FLT * x = (FLT *) calloc(n, sizeof(FLT));
    FLT * y = (FLT *) calloc(n, sizeof(FLT));
    FLT * z = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        x[i] = y[i] = i;
    X(stmv)('N', alpha, A, x, beta, y);
    for (int i = 0; i < n; i++)
        z[i] = beta*i;
    z[0] -= alpha;
    z[n-1] += alpha*n;
    FLT err = X(norm_2arg)(y, z, n)/X(norm_1arg)(z, n);
    printf("Symmetric tridiagonal matrix-vector product \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    for (int i = 0; i < n; i++)
        z[i] = i;
    X(bidiagonal) * B = X(symmetric_tridiagonal_cholesky)(A);
    X(bdmv)('N', B, x);
    X(bdmv)('T', B, x);
    for (int i = 0; i < n; i++)
        z[i] = alpha*x[i] + beta*z[i];
    err = X(norm_2arg)(y, z, n)/X(norm_1arg)(y, n);
    printf("Induced Cholesky bidiagonal matrix-vector product \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    for (int i = 0; i < n; i++)
        z[i] = i;
    X(bdsv)('T', B, x);
    X(bdsv)('N', B, x);
    err = X(norm_2arg)(x, z, n)/X(norm_1arg)(z, n);
    printf("Comparison of matrix-vector solves \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    FLT * V = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        V[i+i*n] = 1;
    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));
    X(symmetric_tridiagonal_eig)(A, V, lambda);
    for (int i = 0; i < n; i++)
        x[i] = i;
    X(gemv)('T', n, n, 1, V, x, 0, z);
    for (int i = 0; i < n; i++)
        z[i] *= lambda[i];
    X(gemv)('N', n, n, 1, V, z, 0, x);
    for (int i = 0; i < n; i++)
        z[i] = i;
    for (int i = 0; i < n; i++)
        z[i] = alpha*x[i] + beta*z[i];
    err = X(norm_2arg)(y, z, n)/X(norm_1arg)(y, n);
    printf("Induced spectral decomposition matrix-vector product \t |%20.2e ", (double) err);
    X(checktest)(err, 2*n, checksum);

    FLT * lambda_true = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambda_true[i] = X(pow)(2*X(__sinpi)((i+ONE(FLT))/(2*n+2)), 2);
    err = X(norm_2arg)(lambda, lambda_true, n)/X(norm_1arg)(lambda_true, n);
    printf("Symmetric tridiagonal eigenvalues \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(symmetric_tridiagonal) * D = (X(symmetric_tridiagonal) *) malloc(sizeof(X(symmetric_tridiagonal)));
    FLT * c = (FLT *) calloc(n, sizeof(FLT));
    FLT * d = (FLT *) calloc(n-1, sizeof(FLT));
    for (int i = 0; i < n; i++)
        c[i] = A->a[i];
    for (int i = 0; i < n-1; i++)
        d[i] = A->b[i];
    D->a = c;
    D->b = d;
    D->n = n;
    X(symmetric_tridiagonal) * C = X(symmetric_tridiagonal_congruence)(A, D, V);
    err = X(pow)(C->a[n-1]-1, 2);
    for (int i = 0; i < n-1; i++)
        err += X(pow)(C->a[i]-1, 2) + X(pow)(C->b[i]-0, 2);
    err = X(sqrt)(err);
    printf("Self-congruence transformations (Ax = λAx) \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    int mu = 40;
    int m = mu-10;
    char PARITY = 'E';
    int shft;
    if (PARITY == 'E') shft = 0;
    else if (PARITY == 'O') shft = 1;
    else shft = -1;
    X(symmetric_tridiagonal) * T = X(create_A_shtsdtev)(n, mu, m, PARITY);
    X(symmetric_tridiagonal) * S = X(create_B_shtsdtev)(n, m, PARITY);
    for (int i = 0; i < n*n; i++)
        V[i] = 0;
    for (int i = 0; i < n; i++)
        V[i+i*n] = 1;
    X(symmetric_definite_tridiagonal_eig)(T, S, V, lambda);
    for (int l = shft; l < 2*n-(mu-m)+shft; l += 2)
        lambda_true[l/2] = (l+mu)*(l+mu+1);
    err = X(norm_2arg)(lambda, lambda_true, n-(mu-m)/2)/X(norm_1arg)(lambda_true, n-(mu-m)/2);
    printf("Symmetric-definite tridiagonal generalized eigenvalues \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    FLT * BV = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * VtBV = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * I = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        I[i+i*n] = 1;
    for (int j = 0; j < n; j++) {
        X(stmv)('N', 1, S, V+j*n, 0, BV+j*n);
        X(gemv)('T', n, n, 1, V, BV+j*n, 0, VtBV+j*n);
    }
    err = X(norm_2arg)(VtBV, I, n*n)/X(norm_1arg)(I, n*n);
    printf("Numerical B-orthogonality of generalized eigenvectors \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(bidiagonal) * R = X(create_R_shtsdtev)(n, m, PARITY);
    for (int j = 0; j < n; j++)
        X(bdmv)('T', R, V+j*n);
    FLT * QtQ = (FLT *) calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        X(gemv)('T', n, n, 1, V, V+j*n, 0, QtQ+j*n);
    err = X(norm_2arg)(QtQ, I, n*(n-(mu-m)/2))/X(norm_1arg)(I, n*(n-(mu-m)/2));
    printf("Numerical RRᵀ-orthogonality of gen. eigenvectors \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_symmetric_tridiagonal)(A);
    X(destroy_symmetric_tridiagonal)(C);
    X(destroy_symmetric_tridiagonal)(D);
    X(destroy_symmetric_tridiagonal)(T);
    X(destroy_symmetric_tridiagonal)(S);
    X(destroy_bidiagonal)(B);
    X(destroy_bidiagonal)(R);
    free(x);
    free(y);
    free(z);
    free(V);
    free(BV);
    free(VtBV);
    free(I);
    free(QtQ);
    free(lambda);
    free(lambda_true);
}
