void Y(test_tridiagonal)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 50;

    X(symmetric_tridiagonal) * A = malloc(sizeof(X(symmetric_tridiagonal)));
    FLT * a = malloc(n*sizeof(FLT));
    FLT * b = malloc((n-1)*sizeof(FLT));
    for (int i = 0; i < n; i++)
        a[i] = 2;
    for (int i = 0; i < n-1; i++)
        b[i] = -1;
    A->a = a;
    A->b = b;
    A->n = n;

    FLT alpha = 2;
    FLT beta = 0.5;
    FLT * x = calloc(n, sizeof(FLT));
    FLT * y = calloc(n, sizeof(FLT));
    FLT * z = calloc(n, sizeof(FLT));
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

    FLT * V = calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        V[i+i*n] = 1;
    FLT * lambda = calloc(n, sizeof(FLT));
    X(symmetric_tridiagonal_eig)(A, V, lambda);
    for (int i = 0; i < n; i++)
        x[i] = i;
    X(gemv)('T', n, n, 1, V, n, x, 0, z);
    for (int i = 0; i < n; i++)
        z[i] *= lambda[i];
    X(gemv)('N', n, n, 1, V, n, z, 0, x);
    for (int i = 0; i < n; i++)
        z[i] = i;
    for (int i = 0; i < n; i++)
        z[i] = alpha*x[i] + beta*z[i];
    err = X(norm_2arg)(y, z, n)/X(norm_1arg)(y, n);
    printf("Induced spectral decomposition matrix-vector product \t |%20.2e ", (double) err);
    X(checktest)(err, 4*n, checksum);

    FLT * lambda_true = calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambda_true[i] = Y(pow)(2*Y(__sinpi)((i+ONE(FLT))/(2*n+2)), 2);
    err = X(norm_2arg)(lambda, lambda_true, n)/X(norm_1arg)(lambda_true, n);
    printf("Symmetric tridiagonal eigenvalues \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(symmetric_tridiagonal) * D = malloc(sizeof(X(symmetric_tridiagonal)));
    FLT * c = malloc(n*sizeof(FLT));
    FLT * d = malloc((n-1)*sizeof(FLT));
    for (int i = 0; i < n; i++)
        c[i] = A->a[i];
    for (int i = 0; i < n-1; i++)
        d[i] = A->b[i];
    D->a = c;
    D->b = d;
    D->n = n;
    X(symmetric_tridiagonal) * C = X(symmetric_tridiagonal_congruence)(A, D, V);
    err = Y(pow)(C->a[n-1]-1, 2);
    for (int i = 0; i < n-1; i++)
        err += Y(pow)(C->a[i]-1, 2) + Y(pow)(C->b[i]-0, 2);
    err = Y(sqrt)(err);
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

    FLT * BV = calloc(n*n, sizeof(FLT));
    FLT * VtBV = calloc(n*n, sizeof(FLT));
    FLT * Id = calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        Id[i+i*n] = 1;
    for (int j = 0; j < n; j++) {
        X(stmv)('N', 1, S, V+j*n, 0, BV+j*n);
        X(gemv)('T', n, n, 1, V, n, BV+j*n, 0, VtBV+j*n);
    }
    err = X(norm_2arg)(VtBV, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical B-orthogonality of generalized eigenvectors \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(bidiagonal) * R = X(create_R_shtsdtev)(n, m, PARITY);
    for (int j = 0; j < n; j++)
        X(bdmv)('T', R, V+j*n);
    FLT * QtQ = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        X(gemv)('T', n, n, 1, V, n, V+j*n, 0, QtQ+j*n);
    err = X(norm_2arg)(QtQ, Id, n*(n-(mu-m)/2))/X(norm_1arg)(Id, n*(n-(mu-m)/2));
    printf("Numerical RRᵀ-orthogonality of gen. eigenvectors \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    for (int i = 0; i < n; i++)
        lambda[n-1-i] = Y(pow)(2*Y(__sinpi)((i+ONE(FLT))/(2*n+2)), 2);
    for (int i = 0; i < n-1; i++)
        A->b[i] = 1;
    X(symmetric_tridiagonal_symmetric_eigen) * F = X(symmetric_tridiagonal_symmetric_eig)(A, lambda, 1);
    for (int j = 0; j < n; j++)
        X(semv)(F, Id+j*n, 1, V+j*n);
    FLT * V2 = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        X(semv)(F, V+j*n, 1, V2+j*n);
    err = X(norm_2arg)(V2, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical symmetric orthogonality w/ known eigenvalues \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(skew_symmetric_tridiagonal) * E = X(create_rectdisk_angular_momentum)(n-1, beta);
    FLT * DE = calloc(n*n, sizeof(FLT));
    FLT * DEtE = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        X(ktmv)('N', 1, E, Id+j*n, 0, DE+j*n);
        X(ktmv)('T', 1, E, DE+j*n, 0, DEtE+j*n);
    }

    X(symmetric_tridiagonal) * E1 = malloc(sizeof(X(symmetric_tridiagonal)));
    E1->a = calloc((n+1)/2, sizeof(FLT));
    E1->b = calloc((n-1)/2, sizeof(FLT));
    E1->n = (n+1)/2;

    X(symmetric_tridiagonal) * E2 = malloc(sizeof(X(symmetric_tridiagonal)));
    E2->a = calloc(n/2, sizeof(FLT));
    E2->b = calloc(n/2-1, sizeof(FLT));
    E2->n = n/2;

    X(skew_to_symmetric_tridiagonal)(E, E1, E2);

    err = 0;
    for (int i = 0; i < E1->n; i++)
        err += (E1->a[i] - DEtE[2*i+2*i*n])*(E1->a[i] - DEtE[2*i+2*i*n]);
    for (int i = 0; i < E1->n-1; i++)
        err += (E1->b[i] - DEtE[2*i+2+2*i*n])*(E1->b[i] - DEtE[2*i+2+2*i*n]);
    for (int i = 0; i < E2->n; i++)
        err += (E2->a[i] - DEtE[2*i+1+(2*i+1)*n])*(E2->a[i] - DEtE[2*i+1+(2*i+1)*n]);
    for (int i = 0; i < E2->n-1; i++)
        err += (E2->b[i] - DEtE[2*i+3+(2*i+1)*n])*(E2->b[i] - DEtE[2*i+3+(2*i+1)*n]);

    printf("Skew-symmetric to symmetric tridiagonal pair AᵀA = B ⊕ C |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_symmetric_tridiagonal)(A);
    X(destroy_symmetric_tridiagonal)(C);
    X(destroy_symmetric_tridiagonal)(D);
    X(destroy_symmetric_tridiagonal)(E1);
    X(destroy_symmetric_tridiagonal)(E2);
    X(destroy_symmetric_tridiagonal)(T);
    X(destroy_symmetric_tridiagonal)(S);
    X(destroy_skew_symmetric_tridiagonal)(E);
    X(destroy_bidiagonal)(B);
    X(destroy_bidiagonal)(R);
    X(destroy_symmetric_tridiagonal_symmetric_eigen)(F);
    free(x);
    free(y);
    free(z);
    free(V);
    free(V2);
    free(BV);
    free(VtBV);
    free(Id);
    free(QtQ);
    free(DE);
    free(DEtE);
    free(lambda);
    free(lambda_true);
}
