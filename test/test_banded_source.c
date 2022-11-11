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

static inline X(banded) * X(create_M_test)(const int m, const int n) {
    X(banded) * M = X(calloc_banded)(m, n, 2, 2);
    for (int j = 0; j < n; j++) {
        X(set_banded_index)(M, 1, j-2, j);
        X(set_banded_index)(M, 2, j-1, j);
        X(set_banded_index)(M, 4, j  , j);
        X(set_banded_index)(M, 2, j+1, j);
        X(set_banded_index)(M, 1, j+2, j);
    }
    return M;
}

void X(inner_test_banded)(int * checksum, int n) {
    int m = n, NTIMES = 10;
    FLT err;
    struct timeval start, end;

    X(banded) * M = X(create_M_test)(m, n);
    X(banded_qr) * QR = X(banded_qrfact)(M);

    FLT * Idm = calloc(m*m, sizeof(FLT));
    FLT * Idn = calloc(n*n, sizeof(FLT));
    FLT * DM = calloc(m*n, sizeof(FLT));
    FLT * DQR = calloc(m*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        DQR[j+j*m] = Idn[j+j*n] = 1;
        X(gbmv)(1, M, Idn+j*n, 0, DM+j*m);
        X(brmv)('N', QR, DQR+j*m);
        X(bqmv)('N', QR, DQR+j*m);
    }
    err = X(norm_2arg)(DM, DQR, m*n)/X(norm_1arg)(DM, m*n);
    printf("Numerical error of ||M - QR||/||M|| \t (%5i×%5i) \t |%20.2e ", m, n, (double) err);
    X(checktest)(err, MAX(m, n), checksum);

    FLT * QtQ = calloc(m*m, sizeof(FLT));
    for (int j = 0; j < m; j++) {
        QtQ[j+j*m] = Idm[j+j*m] = 1;
        X(bqmv)('N', QR, QtQ+j*m);
        X(bqmv)('T', QR, QtQ+j*m);
    }
    err = X(norm_2arg)(QtQ, Idm, m*m)/X(norm_1arg)(Idm, m*m);
    printf("Numerical error of ||QᵀQ - I||/||I|| \t (%5i×%5i) \t |%20.2e ", m, m, (double) err);
    X(checktest)(err, m, checksum);

    FLT * RtR = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        RtR[j+j*n] = 1;
        X(brmv)('N', QR, RtR+j*n);
        X(brmv)('T', QR, RtR+j*n);
    }

    FLT * MtM = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < m; k++)
                MtM[i+j*n] += DM[k+i*m]*DM[k+j*m];

    err = X(norm_2arg)(MtM, RtR, n*n)/X(norm_1arg)(MtM, n*n);
    printf("Numerical error of ||MᵀM - RᵀR||/||MᵀM|| (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    FLT * RinvR = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        RinvR[j+j*n] = 1;
        X(brmv)('N', QR, RinvR+j*n);
        X(brsv)('N', QR, RinvR+j*n);
    }
    err = X(norm_2arg)(RinvR, Idn, n*n)/X(norm_1arg)(Idn, n*n);
    printf("Numerical error of ||R⁻¹R - I||/||I|| \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    FLT * RtinvRt = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        RtinvRt[j+j*n] = 1;
        X(brmv)('T', QR, RtinvRt+j*n);
        X(brsv)('T', QR, RtinvRt+j*n);
    }
    err = X(norm_2arg)(RtinvRt, Idn, n*n)/X(norm_1arg)(Idn, n*n);
    printf("Numerical error of ||R⁻ᵀRᵀ - I||/||I|| \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    X(banded_ql) * QL = X(banded_qlfact)(M);
    FLT * DQL = calloc(m*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        DQL[j+j*m] = 1;
        X(blmv)('N', QL, DQL+j*m);
        X(bqmv)('N', QL, DQL+j*m);
    }
    err = X(norm_2arg)(DM, DQL, m*n)/X(norm_1arg)(DM, m*n);
    printf("Numerical error of ||M - QL||/||M|| \t (%5i×%5i) \t |%20.2e ", m, n, (double) err);
    X(checktest)(err, MAX(m, n), checksum);

    free(QtQ);
    QtQ = calloc(m*m, sizeof(FLT));
    for (int j = 0; j < m; j++) {
        QtQ[j+j*m] = 1;
        X(bqmv)('N', QL, QtQ+j*m);
        X(bqmv)('T', QL, QtQ+j*m);
    }
    err = X(norm_2arg)(QtQ, Idm, m*m)/X(norm_1arg)(Idm, m*m);
    printf("Numerical error of ||QᵀQ - I||/||I|| \t (%5i×%5i) \t |%20.2e ", m, m, (double) err);
    X(checktest)(err, m, checksum);

    FLT * LtL = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        LtL[j+j*n] = 1;
        X(blmv)('N', QL, LtL+j*n);
        X(blmv)('T', QL, LtL+j*n);
    }

    err = X(norm_2arg)(MtM, LtL, n*n)/X(norm_1arg)(MtM, n*n);
    printf("Numerical error of ||MᵀM - LᵀL||/||MᵀM|| (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    FLT * LinvL = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        LinvL[j+j*n] = 1;
        X(blmv)('N', QL, LinvL+j*n);
        X(blsv)('N', QL, LinvL+j*n);
    }
    err = X(norm_2arg)(LinvL, Idn, n*n)/X(norm_1arg)(Idn, n*n);
    printf("Numerical error of ||L⁻¹L - I||/||I|| \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    FLT * LtinvLt = calloc(n*n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        LtinvLt[j+j*n] = 1;
        X(blmv)('T', QL, LtinvLt+j*n);
        X(blsv)('T', QL, LtinvLt+j*n);
    }
    err = X(norm_2arg)(LtinvLt, Idn, n*n)/X(norm_1arg)(Idn, n*n);
    printf("Numerical error of ||L⁻ᵀLᵀ - I||/||I|| \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_banded)(M);
    free(DM);
    free(RtR);

    M = X(create_M_test)(n, n);
    X(banded) * BR = X(create_M_test)(n, n);
    X(banded_cholfact)(BR);
    X(triangular_banded) * R = X(convert_banded_to_triangular_banded)(BR);

    DM = calloc(n*n, sizeof(FLT));
    RtR = calloc(n*n, sizeof(FLT));

    for (int j = 0; j < n; j++) {
        DM[j+j*n] = RtR[j+j*n] = 1;
        X(gbmv)(1, M, Idn+j*n, 0, DM+j*n);
        X(tbmv)('N', R, RtR+j*n);
        X(tbmv)('T', R, RtR+j*n);
    }
    err = X(norm_2arg)(DM, RtR, n*n)/X(norm_1arg)(DM, n*n);
    printf("Numerical error of ||M - RᵀR||/||M|| \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    {
        X(banded) * X = X(create_jacobi_multiplication)(0, n, n, 0, 0);
        int N = MIN(n, 10);
        FLT c[N], A[N], B[N], C[N+1];
        for (int k = 0; k < N; k++) {
            A[k] = (2*k+1.0)/(k+1.0);
            B[k] = 0.0;
            C[k] = k/(k+1.0);
            c[k] = 1.0/(k+1.0);
        }
        X(banded) * M = X(operator_orthogonal_polynomial_clenshaw)(N, c, 1, A, B, C, X, 1);
        FT_TIME(X(operator_orthogonal_polynomial_clenshaw)(N, c, 1, A, B, C, X, 1);, start, end, 1)
        printf("Time for operator OP Clenshaw \t\t (%5i×%5i) \t |%20.6f s\n", n, n, elapsed(&start, &end, 1));
        X(destroy_banded)(M);
        X(destroy_banded)(X);
    }

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
    err = X(norm_2arg)(BinvA, BinvAtrue, n*n)/X(norm_1arg)(BinvA, n*n);
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

    FLT * D = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        D[i] = 1;
    X(tb_eigen_FMM) * F = X(tb_eig_FMM)(A, B, D);
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

    FLT * z = malloc(n*sizeof(FLT));
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

    err = 0;
    FLT alpha = 0.123, beta = 0.456;
    for (int norm = 0; norm < 2; norm++) {
        X(banded) * M1 = X(create_jacobi_multiplication)(norm, n, n+1, alpha, beta);
        X(banded) * M2 = X(create_jacobi_multiplication)(norm, n+1, n, alpha, beta);
        X(banded) * ImM2 = X(calloc_banded)(n, n, 2, 2);
        for (int j = 0; j < n; j++)
            X(set_banded_index)(ImM2, 1, j, j);
        X(gbmm)(-1, M1, M2, 1, ImM2);
        X(banded) * L = X(create_jacobi_lowering)(norm, n, n, alpha, beta);
        X(banded) * R = X(create_jacobi_raising)(norm, n, n, alpha, beta);
        X(banded) * LR = X(calloc_banded)(n, n, 2, 2);
        X(gbmm)(1, L, R, 0, LR);
        err += X(norm_2arg)(ImM2->data, LR->data, 5*n)/X(norm_1arg)(LR->data, 5*n);
        X(destroy_banded)(M1);
        X(destroy_banded)(M2);
        X(destroy_banded)(ImM2);
        X(destroy_banded)(L);
        X(destroy_banded)(R);
        X(destroy_banded)(LR);
    }
    printf("Jacobi lowering*raising vs. 1-x^2 \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, 8, checksum);

    X(banded) * XP = X(create_jacobi_multiplication)(1, n, n, alpha, beta);
    for (int i = 0; i < n-1; i++) {
        X(set_banded_index)(XP, 1-X(get_banded_index)(XP, i, i), i, i);
        X(set_banded_index)(XP, -X(get_banded_index)(XP, i, i+1), i, i+1);
        X(set_banded_index)(XP, -X(get_banded_index)(XP, i+1, i), i+1, i);
    }
    X(set_banded_index)(XP, 1-X(get_banded_index)(XP, n-1, n-1), n-1, n-1);
    X(symmetric_tridiagonal) * JP = X(convert_banded_to_symmetric_tridiagonal)(XP);
    X(symmetric_tridiagonal_qr) * TQR = X(symmetric_tridiagonal_qrfact)(JP);
    FLT * ts = malloc((n-1)*sizeof(FLT));
    FLT * tc = malloc((n-1)*sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        ts[i] = Y(sqrt)(((i+1)*(i+beta+1))/((i+alpha+2)*(i+alpha+beta+2)));
        tc[i] = Y(sqrt)(((alpha+1)*(2*i+alpha+beta+3))/((i+alpha+2)*(i+alpha+beta+2)));
    }
    err = X(norm_2arg)(TQR->s, ts, n-1)/X(norm_1arg)(ts, n-1) + X(norm_2arg)(TQR->c, tc, n-1)/X(norm_1arg)(tc, n-1);
    X(destroy_symmetric_tridiagonal)(JP);
    X(destroy_symmetric_tridiagonal_qr)(TQR);
    printf("Tridiagonal I-X = QR vs. true Q \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, Y(pow)(n, 1.5), checksum);

    err = 0;
    alpha = 0.123;
    for (int norm = 0; norm < 2; norm++) {
        X(banded) * M = X(create_laguerre_multiplication)(norm, n, n, alpha);
        X(banded) * L = X(create_laguerre_lowering)(norm, n, n, alpha);
        X(banded) * R = X(create_laguerre_raising)(norm, n, n, alpha);
        X(banded) * LR = X(calloc_banded)(n, n, 1, 1);
        X(gbmm)(1, L, R, 0, LR);
        err += X(norm_2arg)(M->data, LR->data, 3*n)/X(norm_1arg)(LR->data, 3*n);
        X(destroy_banded)(M);
        X(destroy_banded)(L);
        X(destroy_banded)(R);
        X(destroy_banded)(LR);
    }
    printf("Laguerre lowering*raising vs. x \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, 1, checksum);

    XP = X(create_laguerre_multiplication)(1, n, n, alpha);
    JP = X(convert_banded_to_symmetric_tridiagonal)(XP);
    TQR = X(symmetric_tridiagonal_qrfact)(JP);
    for (int i = 0; i < n-1; i++) {
        ts[i] = Y(sqrt)((i+1)/(i+alpha+2));
        tc[i] = Y(sqrt)((alpha+1)/(i+alpha+2));
    }
    X(banded) * R1 = X(create_laguerre_raising)(1, n, n, alpha);
    X(banded) * R2 = X(create_laguerre_raising)(1, n, n, alpha+1);
    X(banded) * R12 = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(1, R2, R1, 0, R12);
    err = X(norm_2arg)(TQR->s, ts, n-1)/X(norm_1arg)(ts, n-1) + X(norm_2arg)(TQR->c, tc, n-1)/X(norm_1arg)(tc, n-1);
    printf("Tridiagonal   X = QR vs. true Q \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, Y(pow)(n, 1.5), checksum);
    err = X(norm_2arg)(TQR->R->data, R12->data, 3*(n-1))/X(norm_1arg)(R12->data, 3*(n-1));
    printf("Tridiagonal   X = QR vs. true R \t (%5i×%5i) \t |%20.2e ", n, n, (double) err);
    X(checktest)(err, n, checksum);

    X(destroy_banded)(M);
    X(destroy_banded)(R1);
    X(destroy_banded)(R2);
    X(destroy_banded)(R12);
    X(destroy_banded_ql)(QL);
    X(destroy_banded_qr)(QR);
    X(destroy_symmetric_tridiagonal)(JP);
    X(destroy_symmetric_tridiagonal_qr)(TQR);
    X(destroy_triangular_banded)(A);
    X(destroy_triangular_banded)(B);
    X(destroy_triangular_banded)(R);
    X(destroy_tb_eigen_FMM)(F);
    free(BinvA);
    free(BinvAtrue);
    free(AV);
    free(BVL);
    free(D);
    free(DM);
    free(DQL);
    free(DQR);
    free(Idm);
    free(Idn);
    free(LtL);
    free(LinvL);
    free(LtinvLt);
    free(MtM);
    free(QtQ);
    free(RtR);
    free(RinvR);
    free(RtinvRt);
    free(V);
    free(lambda);
    free(ts);
    free(tc);
    free(x);
    free(y);
    free(z);
}

void X(inner_timing_test_banded_FMM)(int * checksum, int n) {
    int NTIMES = 10;
    struct timeval start, end;

    X(triangular_banded) * A = X(create_A_test)(n);
    X(triangular_banded) * B = X(create_B_test)(n);
    FLT * D = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        D[i] = 1;

    printf("Size of a dense matrix \t\t (%7i×%7i) \t |", n, n);
    print_summary_size(sizeof(FLT)*n*n);

    gettimeofday(&start, NULL);
    X(tb_eigen_FMM) * F = X(tb_eig_FMM)(A, B, D);
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
    free(D);
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
        for (int n = 1024; n < 32768; n *= 2) {
            X(inner_timing_test_banded_FMM)(checksum, n);
            X(inner_timing_test_banded_ADI)(checksum, n);
        }
    }
}
