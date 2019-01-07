void X(test_arrow)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 50;

    X(symmetric_arrow) * A = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    A->a = (FLT *) calloc(n-1, sizeof(FLT));
    A->b = (FLT *) calloc(n-1, sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        A->a[i] = (i+1)*(i+1);
        A->b[i] = i+1;
    }
    A->c = (n*n+n-1);
    A->n = n;
    X(upper_arrow) * R = X(symmetric_arrow_cholesky)(A);
    FLT * d_true = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * e_true = (FLT *) calloc(n-1, sizeof(FLT));
    FLT f_true = n;
    for (int i = 0; i < n-1; i++) {
        d_true[i] = i+1;
        e_true[i] = 1;
    }
    FLT err = X(sqrt)(X(pow)(X(norm_2arg)(R->d, d_true, n-1), 2) + X(pow)(X(norm_2arg)(R->e, e_true, n-1), 2) + X(pow)(R->f-f_true, 2))/X(sqrt)(X(pow)(X(norm_1arg)(d_true, n-1), 2) + X(pow)(X(norm_1arg)(e_true, n-1), 2) + X(pow)(f_true, 2));
    printf("Frobenius error in the Cholesky factor \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    FLT alpha = 2;
    FLT beta = 0.5;
    FLT * x = (FLT *) calloc(n, sizeof(FLT));
    FLT * y = (FLT *) calloc(n, sizeof(FLT));
    FLT * z = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        x[i] = y[i] = z[i] = 1/(i+ONE(FLT));
    X(samv)('N', alpha, A, x, beta, y);
    X(uamv)('N', R, x);
    X(uamv)('T', R, x);
    for (int i = 0; i < n; i++)
        z[i] = alpha*x[i] + beta*z[i];
    err = X(norm_2arg)(y, z, n)/X(norm_1arg)(y, n);
    printf("Comparison of matrix-vector products \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    for (int i = 0; i < n; i++)
        y[i] = 1/(i+ONE(FLT));
    X(uasv)('T', R, x);
    X(uasv)('N', R, x);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Comparison of matrix-vector solves \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(upper_arrow) * S = X(upper_arrow_inv)(R);
    for (int i = 0; i < n; i++)
        x[i] = y[i];
    X(uamv)('N', S, x);
    X(uamv)('N', R, x);
    X(uamv)('T', S, x);
    X(uamv)('T', R, x);
    X(uasv)('N', S, x);
    X(uasv)('N', R, x);
    X(uasv)('T', S, x);
    X(uasv)('T', R, x);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Comparison of inverse matrix-vector solves \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(symmetric_arrow) * B = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    B->a = (FLT *) calloc(n-1, sizeof(FLT));
    B->b = (FLT *) calloc(n-1, sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        B->a[i] = (i+1)*(i+1);
        B->b[i] = i+1;
    }
    B->c = (n*n+n-1);
    B->n = n;
    X(symmetric_arrow) * C = X(symmetric_arrow_similarity)(A, B);
    err = X(pow)(C->c-1, 2);
    for (int i = 0; i < n-1; i++)
        err += X(pow)(C->a[i]-1, 2) + X(pow)(C->b[i]-0, 2);
    err = X(sqrt)(err);
    printf("Self-similarity transformations (Ax = λAx) \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    A = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    A->a = (FLT *) calloc(n-1, sizeof(FLT));
    A->b = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        A->a[i] = (i+1)*(i+1);
        A->b[i] = 1;
        lambda[i] = (i+ONE(FLT)/2)*(i+1);
    }
    A->c = n*n;
    A->n = n;
    lambda[n-1] = n*(n-ONE(FLT)/2);
    X(symmetric_arrow) * AS = X(symmetric_arrow_synthesize)(A, lambda);
    int * p = (int *) malloc(n*sizeof(int));
    int ib = X(symmetric_arrow_process)(AS, p);
    free(p);
    FLT * mu = X(symmetric_arrow_eigvals)(AS, ib);
    err = X(norm_2arg)(lambda, mu, n)/X(norm_1arg)(lambda, n);
    printf("Synthetic symmetric arrow eigenvalues \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(symmetric_arrow_eigen) * F = X(symmetric_arrow_eig)(A);
    FLT * Q = F->Q;
    FLT * QtQ = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * I = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        I[i+i*n] = 1;
    for (int j = 0; j < n; j++)
        X(gemv)('T', n, n, 1, Q, Q+j*n, 0, QtQ+j*n);
    err = X(norm_2arg)(QtQ, I, n*n)/X(norm_1arg)(I, n*n);
    printf("Numerical orthogonality of eigenvectors \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_symmetric_arrow)(A);
    X(destroy_symmetric_arrow)(AS);
    X(destroy_symmetric_arrow)(B);
    X(destroy_symmetric_arrow)(C);
    X(destroy_symmetric_arrow_eigen)(F);
    X(destroy_upper_arrow)(R);
    X(destroy_upper_arrow)(S);
    free(d_true);
    free(e_true);
    free(x);
    free(y);
    free(z);
    free(lambda);
    free(mu);
    free(QtQ);
    free(I);
}
