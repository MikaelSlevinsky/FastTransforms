static inline X(triangular_banded) * X(create_A_test)(const int n) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X(set_triangular_banded_index)(A, 2, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, -i*(i-ONE(FLT)), i-2, i);
        X(set_triangular_banded_index)(A, i*(i+ONE(FLT)), i, i);
    }
    return A;
}

static inline X(triangular_banded) * X(create_B_test)(const int n) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(B, 2, 0, 0);
    if (n > 1)
        X(set_triangular_banded_index)(B, 1, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(B, -1, i-2, i);
        X(set_triangular_banded_index)(B, 1, i, i);
    }
    return B;
}

void X(inner_test_banded)(int * checksum, int n) {
    int NTIMES = 10;
    struct timeval start, end;

    X(triangular_banded) * A = X(create_A_test)(n);
    X(triangular_banded) * B = X(create_B_test)(n);

    FLT * BinvA = calloc(n*n, sizeof(FLT));
    FLT * BinvAtrue = calloc(n*n, sizeof(FLT));
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
    printf("Matrix-vector products & solves \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    FLT * V = calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        V[i+i*n] = 1;
    FLT * lambda = malloc(n*sizeof(FLT));

    X(triangular_banded_eigenvalues)(A, B, lambda);
    X(triangular_banded_eigenvectors)(A, B, V);

    FLT * AV = malloc(n*n*sizeof(FLT));
    FLT * BVL = malloc(n*n*sizeof(FLT));

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            AV[i+j*n] = V[i+j*n];
            BVL[i+j*n] = V[i+j*n]*lambda[j];
        }
        X(tbmv)('N', A, AV+j*n);
        X(tbmv)('N', B, BVL+j*n);
    }

    err = X(norm_2arg)(AV, BVL, n*n)/X(norm_1arg)(lambda, n);
    printf("Numerical error of tb ||AV - BVΛ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, 4, checksum);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            BVL[i+j*n] = V[i+j*n]*lambda[j];
        X(tbsv)('N', B, AV+j*n);
    }

    err = X(norm_2arg)(AV, BVL, n*n)/X(norm_1arg)(lambda, n);
    printf("Numerical error of tb ||B⁻¹AV - VΛ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    FLT * x = malloc(n*sizeof(FLT));
    FLT * y = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        y[i] = x[i] = ONE(FLT)/(i+1);
    X(trmv)('N', n, V, n, x);
    X(trsv)('N', n, V, n, x);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    X(trmv)('T', n, V, n, x);
    X(trsv)('T', n, V, n, x);
    err += X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Numerical error of triangular linear algebra \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(tb_eigen_FMM) * F = X(tb_eig_FMM)(A, B);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            V[i+j*n] = 0;
        V[j+j*n] = 1;
        X(bfmv)('N', F, V+j*n);
        for (int i = 0; i < n; i++) {
            AV[i+j*n] = V[i+j*n];
            BVL[i+j*n] = V[i+j*n]*F->lambda[j];
        }
        X(tbmv)('N', A, AV+j*n);
        X(tbmv)('N', B, BVL+j*n);
    }

    err = X(norm_2arg)(AV, BVL, n*n)/X(norm_1arg)(F->lambda, n);
    printf("Numerical error of FMM'ed tb ||AV - BVΛ|| / ||Λ|| \t |%20.2e ", (double) err);
    X(checktest)(err, 4, checksum);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            BVL[i+j*n] = V[i+j*n]*F->lambda[j];
        X(tbsv)('N', B, AV+j*n);
    }

    err = X(norm_2arg)(AV, BVL, n*n)/X(norm_1arg)(F->lambda, n);
    printf("Numerical error of FMM'ed tb ||B⁻¹AV - VΛ|| / ||Λ|| \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    printf("Size of a dense matrix \t\t\t (%5i×%5i) \t |", n, n);
    print_summary_size(sizeof(FLT)*n*n);

    printf("Size of the triangular banded eigendecomposition \t |");
    print_summary_size(X(summary_size_tb_eigen_FMM)(F));

    FT_TIME({X(bfmv)('N', F, x); X(bfsv)('N', F, x);}, start, end, NTIMES)
    printf("Time for fwd-bckwd solves \t\t (%5i×%5i) \t |%20.6f s\n", n, n, elapsed(&start, &end, NTIMES));

    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Error of fwd-bckwd solves \t\t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    FLT * z  = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        z[i] = ONE(FLT)/(i+1);
    for (int i = 0; i < n; i++)
        x[i] *= z[i]/2;
    X(bfmv)('N', F, x);
    X(scale_columns_tb_eigen_FMM)(0.5, z, F);
    X(bfmv)('N', F, y);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(x, n);

    X(bfmv)('N', F, x);
    X(scale_rows_tb_eigen_FMM)(2, z, F);
    X(bfmv)('N', F, y);
    for (int i = 0; i < n; i++)
        x[i] *= 2*z[i];
    err += X(norm_2arg)(x, y, n)/X(norm_1arg)(x, n);

    printf("Check row/column scalings \t\t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_triangular_banded)(A);
    X(destroy_triangular_banded)(B);
    X(destroy_tb_eigen_FMM)(F);
    free(BinvA);
    free(BinvAtrue);
    free(AV);
    free(BVL);
    free(V);
    free(lambda);
    free(x);
    free(y);
    free(z);
}

void X(inner_timing_test_banded_FMM)(int * checksum, int n) {
    int NTIMES = 10;
    struct timeval start, end;

    X(triangular_banded) * A = X(create_A_test)(n);
    X(triangular_banded) * B = X(create_B_test)(n);

    printf("Size of a dense matrix \t\t (%7i×%7i) \t |", n, n);
    print_summary_size(sizeof(FLT)*n*n);

    gettimeofday(&start, NULL);
    X(tb_eigen_FMM) * F = X(tb_eig_FMM)(A, B);
    gettimeofday(&end, NULL);

    printf("Size of the triangular banded eigendecomposition (FMM) \t |");
    print_summary_size(X(summary_size_tb_eigen_FMM)(F));

    printf("Time for factorization \t\t (%7i×%7i) \t |%20.6f s\n", n, n, elapsed(&start, &end, 1));

    FLT * x = malloc(n*sizeof(FLT));
    FLT * y = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        y[i] = x[i] = ONE(FLT)/(i+1);
    FT_TIME({X(bfmv)('N', F, x); X(bfsv)('N', F, x);}, start, end, NTIMES)
    printf("Time for fwd-bckwd solves \t (%7i×%7i) \t |%20.6f s\n", n, n, elapsed(&start, &end, NTIMES));

    FLT err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Error of fwd-bckwd solves \t (%7i×%7i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_triangular_banded)(A);
    X(destroy_triangular_banded)(B);
    X(destroy_tb_eigen_FMM)(F);
    free(x);
    free(y);
}

void X(inner_timing_test_banded_ADI)(int * checksum, int n) {
    int NTIMES = 10;
    struct timeval start, end;

    X(triangular_banded) * A = X(create_A_test)(n);
    X(triangular_banded) * B = X(create_B_test)(n);

    printf("Size of a dense matrix \t\t (%7i×%7i) \t |", n, n);
    print_summary_size(sizeof(FLT)*n*n);

    gettimeofday(&start, NULL);
    X(tb_eigen_ADI) * F = X(tb_eig_ADI)(A, B);
    gettimeofday(&end, NULL);

    printf("Size of the triangular banded eigendecomposition (ADI) \t |");
    print_summary_size(X(summary_size_tb_eigen_ADI)(F));

    printf("2-norm estimate of triangular banded eigenvectors \t |%20.6f \n", (double) X(normest_tb_eigen_ADI)(F));

    printf("Time for factorization \t\t (%7i×%7i) \t |%20.6f s\n", n, n, elapsed(&start, &end, 1));

    FLT * x = malloc(n*sizeof(FLT));
    FLT * y = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        y[i] = x[i] = ONE(FLT)/(i+1);
    FT_TIME({X(bfmv_ADI)('N', F, x); X(bfsv_ADI)('N', F, x);}, start, end, NTIMES)
    printf("Time for fwd-bckwd solves \t (%7i×%7i) \t |%20.6f s\n", n, n, elapsed(&start, &end, NTIMES));

    FLT err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Error of fwd-bckwd solves \t (%7i×%7i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_triangular_banded)(A);
    X(destroy_triangular_banded)(B);
    X(destroy_tb_eigen_ADI)(F);
    free(x);
    free(y);
}


void Y(test_banded)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int nmin = 256, nmax = 1024;

    for (int n = nmin; n < nmax; n *= 2)
        X(inner_test_banded)(checksum, n);
    if (sizeof(FLT) == sizeof(double)) {
        printf("\n\n");
        for (int n = 1024; n < 131072; n *= 2) {
            X(inner_timing_test_banded_FMM)(checksum, n);
            X(inner_timing_test_banded_ADI)(checksum, n);
        }
    }
}
