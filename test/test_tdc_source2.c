void X(inner_test_tdc_drop_precision)(int * checksum, int n) {
    struct timeval start, end;

    printf("Size of a dense matrix \t\t\t (%5i×%5i) \t |", n, n);
    print_summary_size(sizeof(FLT)*n*n);

    int mu = n/2;
    int m = 0;
    char PARITY = 'E';
    int shft;
    if (PARITY == 'E') shft = 0;
    else if (PARITY == 'O') shft = 1;
    else shft = -1;
    X(symmetric_tridiagonal) * T = X(create_A_shtsdtev)(n, mu, m, PARITY);
    X(symmetric_tridiagonal) * S = X(create_B_shtsdtev)(n, m, PARITY);
    X2(symmetric_tridiagonal) * T2 = X2(create_A_shtsdtev)(n, mu, m, PARITY);
    X2(symmetric_tridiagonal) * S2 = X2(create_B_shtsdtev)(n, m, PARITY);

    FLT * Id = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        Id[i+i*n] = 1;

    FLT * V = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * VtBV = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * VtAV = (FLT *) calloc(n*n, sizeof(FLT));

    gettimeofday(&start, NULL);
    X2(tdc_eigen) * F2 = X2(sdtdc_eig)(T2, S2);
    gettimeofday(&end, NULL);
    printf("Time for extended precision eigensolve \t\t\t |%20.6f s\n", elapsed(&start, &end, 1));

    gettimeofday(&start, NULL);
    X(tdc_eigen) * F = X(drop_precision_tdc_eigen)(F2);
    gettimeofday(&end, NULL);
    printf("Time to drop precision in the factorization \t\t |%20.6f s\n", elapsed(&start, &end, 1));
    X2(destroy_tdc_eigen)(F2);

    gettimeofday(&start, NULL);
    for (int j = 0; j < n; j++)
        X(tdmv)('N', 1, F, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(stmv)('N', 1, S, V+j*n, 0, VtBV+j*n);
    for (int i = 0; i < n*n; i++)
        V[i] = VtBV[i];
    for (int j = 0; j < n; j++)
        X(tdmv)('T', 1, F, V+j*n, 0, VtBV+j*n);
    for (int j = 0; j < n; j++)
        X(tdmv)('N', 1, F, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(stmv)('N', 1, T, V+j*n, 0, VtAV+j*n);
    for (int i = 0; i < n*n; i++)
        V[i] = VtAV[i];
    for (int j = 0; j < n; j++)
        X(tdmv)('T', 1, F, V+j*n, 0, VtAV+j*n);
    for (int j = 0; j < n; j++) {
        VtAV[j+j*n] -= F->F0->lambda[j];
        VtBV[j+j*n] -= 1;
    }
    gettimeofday(&end, NULL);
    printf("Time to execute tests \t\t\t\t\t |%20.6f s\n", elapsed(&start, &end, 1));

    printf("Size of the divide-and-conquer eigendecomposition \t |");
    print_summary_size(X(summary_size_tdc_eigen)(F));

    FLT err = X(norm_1arg)(VtBV, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in tdc ||VᵀBV - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(F->F0->lambda, n);
    printf("Numerical error in tdc ||VᵀAV - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);


    gettimeofday(&start, NULL);
    X2(tdc_eigen_FMM) * HF2 = X2(sdtdc_eig_FMM)(T2, S2);
    gettimeofday(&end, NULL);
    printf("Time for extended precision FMM eigensolve \t\t |%20.6f s\n", elapsed(&start, &end, 1));

    gettimeofday(&start, NULL);
    X(tdc_eigen_FMM) * HF = X(drop_precision_tdc_eigen_FMM)(HF2);
    gettimeofday(&end, NULL);
    printf("Time to drop precision in the FMM factorization \t |%20.6f s\n", elapsed(&start, &end, 1));
    X2(destroy_tdc_eigen_FMM)(HF2);

    gettimeofday(&start, NULL);
    for (int j = 0; j < n; j++)
        X(tfmv)('N', 1, HF, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(stmv)('N', 1, S, V+j*n, 0, VtBV+j*n);
    for (int i = 0; i < n*n; i++)
        V[i] = VtBV[i];
    for (int j = 0; j < n; j++)
        X(tfmv)('T', 1, HF, V+j*n, 0, VtBV+j*n);
    for (int j = 0; j < n; j++)
        X(tfmv)('N', 1, HF, Id+j*n, 0, V+j*n);
    for (int j = 0; j < n; j++)
        X(stmv)('N', 1, T, V+j*n, 0, VtAV+j*n);
    for (int i = 0; i < n*n; i++)
        V[i] = VtAV[i];
    for (int j = 0; j < n; j++)
        X(tfmv)('T', 1, HF, V+j*n, 0, VtAV+j*n);
    for (int j = 0; j < n; j++) {
        VtAV[j+j*n] -= HF->F0->lambda[j];
        VtBV[j+j*n] -= 1;
    }
    gettimeofday(&end, NULL);
    printf("Time to execute FMM tests \t\t\t\t |%20.6f s\n", elapsed(&start, &end, 1));

    printf("Size of the FMM'ed eigendecomposition \t\t\t |");
    print_summary_size(X(summary_size_tdc_eigen_FMM)(HF));

    err = X(norm_1arg)(VtBV, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in FMM'ed tdc ||VᵀBV - I|| / ||I|| \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(HF->F0->lambda, n);
    printf("Numerical error in FMM'ed tdc ||VᵀAV - Λ|| / ||Λ|| \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_tridiagonal)(T);
    X(destroy_symmetric_tridiagonal)(S);
    X2(destroy_symmetric_tridiagonal)(T2);
    X2(destroy_symmetric_tridiagonal)(S2);
    X(destroy_tdc_eigen)(F);
    X(destroy_tdc_eigen_FMM)(HF);
    free(Id);
    free(V);
    free(VtAV);
    free(VtBV);
}

void Y(test_tdc_drop_precision)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int nmin = 1024, nmax = 4096;

    for (int n = nmin; n < nmax; n *= 2)
        X(inner_test_tdc_drop_precision)(checksum, n);
}
