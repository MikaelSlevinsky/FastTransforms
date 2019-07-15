void Y(test_triangular_banded)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 256, b = 2;
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, b);
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, b);
    X(set_triangular_banded_index)(B, 2, 0, 0);
    for (int i = 1; i < n; i++) {
        X(set_triangular_banded_index)(A, i*(i+1), i, i);
        X(set_triangular_banded_index)(B, 1, i, i);
    }
    for (int i = 0; i < n-2; i++) {
        X(set_triangular_banded_index)(A, -(i+1)*(i+2), i, i+2);
        X(set_triangular_banded_index)(B, -1, i, i+2);
    }

    FLT * BinvA = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * BinvAtrue = (FLT *) calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        BinvA[j+j*n] = 1;
        X(tbmv)('N', A, BinvA+j*n);
        X(tbsv)('N', B, BinvA+j*n);
        BinvAtrue[j+j*n] = j*(j+1);
        for (int i = j-2; i > 0; i -= 2)
            BinvAtrue[i+j*n] = 2*j;
        if (j%2 == 0)
            BinvAtrue[j*n] = j;

    }
    FLT err = X(norm_2arg)(BinvA, BinvAtrue, n*n)/X(norm_1arg)(BinvA, n*n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            BinvA[i+j*n] = 0;
            BinvAtrue[i+j*n] = 0;
        }
        BinvA[j+j*n] = 1;
        X(tbmv)('T', A, BinvA+j*n);
        X(tbsv)('T', B, BinvA+j*n);
        BinvAtrue[j+j*n] = j*(j+1);
        for (int i = j+2; i < n; i += 2)
            BinvAtrue[i+j*n] = -(2*j+2);
    }
    err += X(norm_2arg)(BinvA, BinvAtrue, n*n)/X(norm_1arg)(BinvA, n*n);
    printf("Forward and inverse matrix-vector products \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);


    FLT * V = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        V[i+i*n] = 1;
    FLT * lambda = (FLT *) malloc(n*sizeof(FLT));

    X(triangular_banded_eigenvalues)(A, B, lambda);
    X(triangular_banded_eigenvectors)(A, B, V);

    FLT * AV = (FLT *) malloc(n*n*sizeof(FLT));
    FLT * BVL = (FLT *) malloc(n*n*sizeof(FLT));

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            AV[i+j*n] = V[i+j*n];
            BVL[i+j*n] = V[i+j*n]*lambda[j];
        }
        X(tbmv)('N', A, AV+j*n);
        X(tbmv)('N', B, BVL+j*n);
    }

    err = X(norm_2arg)(AV, BVL, n*n)/X(norm_1arg)(lambda, n);
    printf("Numerical error of eigensolve ||AV - BVΛ|| / ||Λ|| \t |%20.2e ", (double) err);
    X(checktest)(err, 4, checksum);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            BVL[i+j*n] = V[i+j*n]*lambda[j];
        X(tbsv)('N', B, AV+j*n);
    }

    err = X(norm_2arg)(AV, BVL, n*n)/X(norm_1arg)(lambda, n);
    printf("Numerical error of eigensolve ||B⁻¹AV - VΛ|| / ||Λ|| \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_triangular_banded)(A);
    X(destroy_triangular_banded)(B);
    free(BinvA);
    free(BinvAtrue);
    free(AV);
    free(BVL);
    free(V);
    free(lambda);
}
