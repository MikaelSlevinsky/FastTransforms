void Y(test_tdc)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 256;
    struct timeval start, end;

    int mu = n/2;
    int m = 0;
    char PARITY = 'E';
    int shft;
    if (PARITY == 'E') shft = 0;
    else if (PARITY == 'O') shft = 1;
    else shft = -1;
    X(symmetric_tridiagonal) * T = X(create_A_shtsdtev)(n, mu, m, PARITY);
    X(symmetric_tridiagonal) * S = X(create_B_shtsdtev)(n, m, PARITY);
    FLT * V = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        V[i+i*n] = 1;
    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));
    FLT * lambda_true = (FLT *) calloc(n, sizeof(FLT));
    X(symmetric_definite_tridiagonal_eig)(T, S, V, lambda);
    for (int l = shft; l < 2*n-(mu-m)+shft; l += 2)
        lambda_true[l/2] = (l+mu)*(l+mu+1);
    FLT err = X(norm_2arg)(lambda, lambda_true, n-(mu-m)/2)/X(norm_1arg)(lambda_true, n-(mu-m)/2);
    printf("Symmetric-definite tridiagonal generalized eigenvalues \t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    FLT * Id = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * AV = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * BV = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * VtBV = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * VtAV = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        Id[i+i*n] = 1;
    for (int j = 0; j < n; j++) {
        X(stmv)('N', 1, T, V+j*n, 0, AV+j*n);
        X(stmv)('N', 1, S, V+j*n, 0, BV+j*n);
        X(gemv)('T', n, n, 1, V, AV+j*n, 0, VtAV+j*n);
        X(gemv)('T', n, n, 1, V, BV+j*n, 0, VtBV+j*n);
        VtAV[j+j*n] -= lambda[j];
        VtBV[j+j*n] -= 1;
    }

    err = X(norm_1arg)(VtBV, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in direct ||VᵀBV - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(lambda, n);
    printf("Numerical error in direct ||VᵀAV - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(tdc_eigen) * F = X(sdtdc_eig)(T, S);

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

    err = X(norm_1arg)(VtBV, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in tdc ||VᵀBV - I|| / ||I|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(F->F0->lambda, n);
    printf("Numerical error in tdc ||VᵀAV - Λ|| / ||Λ|| \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(tdc_eigen_FMM) * HF = X(sdtdc_eig_FMM)(T, S);

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

    err = X(norm_1arg)(VtBV, n*n)/X(norm_1arg)(Id, n*n);
    printf("Numerical error in FMM'ed tdc ||VᵀBV - I|| / ||I|| \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    err = X(norm_1arg)(VtAV, n*n)/X(norm_1arg)(HF->F0->lambda, n);
    printf("Numerical error in FMM'ed tdc ||VᵀAV - Λ|| / ||Λ|| \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);

    X(destroy_symmetric_tridiagonal)(T);
    X(destroy_symmetric_tridiagonal)(S);
    X(destroy_tdc_eigen)(F);
    X(destroy_tdc_eigen_FMM)(HF);
    free(V);
    free(AV);
    free(BV);
    free(VtAV);
    free(VtBV);
    free(Id);
    free(lambda);
    free(lambda_true);
}
