X(tb_eigen_FMM) * X(plan_legendre_to_chebyshev)(const int normleg, const int normcheb, const int n) {
    X2(triangular_banded) * A = X2(create_A_legendre_to_chebyshev)(normcheb, n);
    X2(triangular_banded) * B = X2(create_B_legendre_to_chebyshev)(normcheb, n);
    FLT2 * D = malloc(n*sizeof(FLT2));
    X2(create_legendre_to_chebyshev_diagonal_connection_coefficient)(normleg, normcheb, n, D, 1);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B, D);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(tb_eigen_FMM) * X(plan_chebyshev_to_legendre)(const int normcheb, const int normleg, const int n) {
    X2(triangular_banded) * A = X2(create_A_chebyshev_to_legendre)(normleg, n);
    X2(triangular_banded) * B = X2(create_B_chebyshev_to_legendre)(normleg, n);
    FLT2 * D = malloc(n*sizeof(FLT2));
    X2(create_chebyshev_to_legendre_diagonal_connection_coefficient)(normcheb, normleg, n, D, 1);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B, D);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(tb_eigen_FMM) * X(plan_ultraspherical_to_ultraspherical)(const int norm1, const int norm2, const int n, const FLT lambda, const FLT mu) {
    X2(triangular_banded) * A = X2(create_A_ultraspherical_to_ultraspherical)(norm2, n, lambda, mu);
    X2(triangular_banded) * B = X2(create_B_ultraspherical_to_ultraspherical)(norm2, n, mu);
    FLT2 * D = malloc(n*sizeof(FLT2));
    X2(create_ultraspherical_to_ultraspherical_diagonal_connection_coefficient)(norm1, norm2, n, lambda, mu, D, 1);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B, D);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(tb_eigen_FMM) * X(plan_jacobi_to_jacobi)(const int norm1, const int norm2, const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X2(triangular_banded) * A = X2(create_A_jacobi_to_jacobi)(norm2, n, alpha, beta, gamma, delta);
    X2(triangular_banded) * B = X2(create_B_jacobi_to_jacobi)(norm2, n, gamma, delta);
    FLT2 * D = malloc(n*sizeof(FLT2));
    X2(create_jacobi_to_jacobi_diagonal_connection_coefficient)(norm1, norm2, n, alpha, beta, gamma, delta, D, 1);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B, D);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(tb_eigen_FMM) * X(plan_laguerre_to_laguerre)(const int norm1, const int norm2, const int n, const FLT alpha, const FLT beta) {
    X2(triangular_banded) * A = X2(create_A_laguerre_to_laguerre)(norm2, n, alpha, beta);
    X2(triangular_banded) * B = X2(create_B_laguerre_to_laguerre)(norm2, n, beta);
    FLT2 * D = malloc(n*sizeof(FLT2));
    X2(create_laguerre_to_laguerre_diagonal_connection_coefficient)(norm1, norm2, n, alpha, beta, D, 1);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B, D);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(tb_eigen_FMM) * X(plan_jacobi_to_ultraspherical)(const int normjac, const int normultra, const int n, const FLT alpha, const FLT beta, const FLT lambda) {
    X(tb_eigen_FMM) * F = X(plan_jacobi_to_jacobi)(normjac, normultra, n, alpha, beta, lambda-0.5, lambda-0.5);
    if (normultra == 0) {
        FLT * sclrow = malloc(n*sizeof(FLT));
        if (n > 0)
            sclrow[0] = 1;
        for (int i = 1; i < n; i++)
            sclrow[i] = (lambda+i-0.5)/(2*lambda+i-1)*sclrow[i-1];
        X(scale_rows_tb_eigen_FMM)(1, sclrow, F);
        free(sclrow);
    }
    return F;
}

X(tb_eigen_FMM) * X(plan_ultraspherical_to_jacobi)(const int normultra, const int normjac, const int n, const FLT lambda, const FLT alpha, const FLT beta) {
    X(tb_eigen_FMM) * F = X(plan_jacobi_to_jacobi)(normultra, normjac, n, lambda-0.5, lambda-0.5, alpha, beta);
    if (normultra == 0) {
        FLT * sclcol = malloc(n*sizeof(FLT));
        if (n > 0)
            sclcol[0] = 1;
        for (int i = 1; i < n; i++)
            sclcol[i] = (2*lambda+i-1)/(lambda+i-0.5)*sclcol[i-1];
        X(scale_columns_tb_eigen_FMM)(1, sclcol, F);
        free(sclcol);
    }
    return F;
}

X(tb_eigen_FMM) * X(plan_jacobi_to_chebyshev)(const int normjac, const int normcheb, const int n, const FLT alpha, const FLT beta) {
    X(tb_eigen_FMM) * F = X(plan_jacobi_to_jacobi)(normjac, 1, n, alpha, beta, -0.5, -0.5);
    if (normcheb == 0) {
        FLT * sclrow = malloc(n*sizeof(FLT));
        FLT sqrt_1_pi = 1/Y2(tgamma)(0.5);
        FLT sqrt_2_pi = Y2(sqrt)(2)/Y2(tgamma)(0.5);
        if (n > 0)
            sclrow[0] = sqrt_1_pi;
        for (int i = 1; i < n; i++)
            sclrow[i] = sqrt_2_pi;
        X(scale_rows_tb_eigen_FMM)(1, sclrow, F);
        free(sclrow);
    }
    return F;
}

X(tb_eigen_FMM) * X(plan_chebyshev_to_jacobi)(const int normcheb, const int normjac, const int n, const FLT alpha, const FLT beta) {
    X(tb_eigen_FMM) * F = X(plan_jacobi_to_jacobi)(1, normjac, n, -0.5, -0.5, alpha, beta);
    if (normcheb == 0) {
        FLT * sclcol = malloc(n*sizeof(FLT));
        FLT2 sqrtpi = Y2(tgamma)(0.5);
        FLT2 sqrtpi2 = sqrtpi/Y2(sqrt)(2);
        if (n > 0)
            sclcol[0] = sqrtpi;
        for (int i = 1; i < n; i++)
            sclcol[i] = sqrtpi2;
        X(scale_columns_tb_eigen_FMM)(1, sclcol, F);
        free(sclcol);
    }
    return F;
}

X(tb_eigen_FMM) * X(plan_ultraspherical_to_chebyshev)(const int normultra, const int normcheb, const int n, const FLT lambda) {
    X(tb_eigen_FMM) * F = X(plan_ultraspherical_to_jacobi)(normultra, 1, n, lambda, -0.5, -0.5);
    if (normcheb == 0) {
        FLT * sclrow = malloc(n*sizeof(FLT));
        FLT sqrt_1_pi = 1/Y2(tgamma)(0.5);
        FLT sqrt_2_pi = Y2(sqrt)(2)/Y2(tgamma)(0.5);
        if (n > 0)
            sclrow[0] = sqrt_1_pi;
        for (int i = 1; i < n; i++)
            sclrow[i] = sqrt_2_pi;
        X(scale_rows_tb_eigen_FMM)(1, sclrow, F);
        free(sclrow);
    }
    return F;
}

X(tb_eigen_FMM) * X(plan_chebyshev_to_ultraspherical)(const int normcheb, const int normultra, const int n, const FLT lambda) {
    X(tb_eigen_FMM) * F = X(plan_jacobi_to_ultraspherical)(1, normultra, n, -0.5, -0.5, lambda);
    if (normcheb == 0) {
        FLT * sclcol = malloc(n*sizeof(FLT));
        FLT2 sqrtpi = Y2(tgamma)(0.5);
        FLT2 sqrtpi2 = sqrtpi/Y2(sqrt)(2);
        if (n > 0)
            sclcol[0] = sqrtpi;
        for (int i = 1; i < n; i++)
            sclcol[i] = sqrtpi2;
        X(scale_columns_tb_eigen_FMM)(1, sclcol, F);
        free(sclcol);
    }
    return F;
}

X(btb_eigen_FMM) * X(plan_associated_jacobi_to_jacobi)(const int norm1, const int norm2, const int n, const int c, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X2(triangular_banded) * dataA[2][2] = {X2(create_A_associated_jacobi_to_jacobi)(norm2, n, c, alpha, beta, gamma, delta), X2(create_B_associated_jacobi_to_jacobi)(norm2, n, gamma, delta), X2(calloc_triangular_banded)(n, 4), X2(create_I_triangular_banded)(n, 4)};
    X2(triangular_banded) * dataB[2][2] = {X2(calloc_triangular_banded)(n, 4), X2(create_C_associated_jacobi_to_jacobi)(norm2, n, gamma, delta), X2(create_I_triangular_banded)(n, 4), X2(calloc_triangular_banded)(n, 4)};
    X2(block_2x2_triangular_banded) * A = X2(create_block_2x2_triangular_banded)(dataA);
    X2(block_2x2_triangular_banded) * B = X2(create_block_2x2_triangular_banded)(dataB);
    FLT2 * D = malloc(2*n*sizeof(FLT2));
    X2(create_associated_jacobi_to_jacobi_diagonal_connection_coefficient)(norm1, norm2, n, c, alpha, beta, gamma, delta, D, 2);
    X2(create_associated_jacobi_to_jacobi_diagonal_connection_coefficient)(norm1, norm2, n, c, alpha, beta, gamma, delta, D+1, 2);
    X2(btb_eigen_FMM) * F2 = X2(btb_eig_FMM)(A, B, D);
    X(btb_eigen_FMM) * F = X(drop_precision_btb_eigen_FMM)(F2);
    X2(destroy_block_2x2_triangular_banded)(A);
    X2(destroy_block_2x2_triangular_banded)(B);
    X2(destroy_btb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(btb_eigen_FMM) * X(plan_associated_laguerre_to_laguerre)(const int norm1, const int norm2, const int n, const int c, const FLT alpha, const FLT beta) {
    X2(triangular_banded) * dataA[2][2] = {X2(create_A_associated_laguerre_to_laguerre)(norm2, n, c, alpha, beta), X2(create_B_associated_laguerre_to_laguerre)(norm2, n, beta), X2(calloc_triangular_banded)(n, 4), X2(create_I_triangular_banded)(n, 4)};
    X2(triangular_banded) * dataB[2][2] = {X2(calloc_triangular_banded)(n, 2), X2(create_C_associated_laguerre_to_laguerre)(norm2, n, beta), X2(create_I_triangular_banded)(n, 2), X2(calloc_triangular_banded)(n, 2)};
    X2(block_2x2_triangular_banded) * A = X2(create_block_2x2_triangular_banded)(dataA);
    X2(block_2x2_triangular_banded) * B = X2(create_block_2x2_triangular_banded)(dataB);
    FLT2 * D = malloc(2*n*sizeof(FLT2));
    X2(create_associated_laguerre_to_laguerre_diagonal_connection_coefficient)(norm1, norm2, n, c, alpha, beta, D, 2);
    X2(create_associated_laguerre_to_laguerre_diagonal_connection_coefficient)(norm1, norm2, n, c, alpha, beta, D+1, 2);
    X2(btb_eigen_FMM) * F2 = X2(btb_eig_FMM)(A, B, D);
    X(btb_eigen_FMM) * F = X(drop_precision_btb_eigen_FMM)(F2);
    X2(destroy_block_2x2_triangular_banded)(A);
    X2(destroy_block_2x2_triangular_banded)(B);
    X2(destroy_btb_eigen_FMM)(F2);
    free(D);
    return F;
}

X(btb_eigen_FMM) * X(plan_associated_hermite_to_hermite)(const int norm1, const int norm2, const int n, const int c) {
    X2(triangular_banded) * dataA[2][2] = {X2(create_A_associated_hermite_to_hermite)(norm2, n, c), X2(create_B_associated_hermite_to_hermite)(norm2, n), X2(calloc_triangular_banded)(n, 4), X2(create_I_triangular_banded)(n, 4)};
    X2(triangular_banded) * dataB[2][2] = {X2(calloc_triangular_banded)(n, 0), X2(create_C_associated_hermite_to_hermite)(n), X2(create_I_triangular_banded)(n, 0), X2(calloc_triangular_banded)(n, 0)};
    X2(block_2x2_triangular_banded) * A = X2(create_block_2x2_triangular_banded)(dataA);
    X2(block_2x2_triangular_banded) * B = X2(create_block_2x2_triangular_banded)(dataB);
    FLT2 * D = malloc(2*n*sizeof(FLT2));
    X2(create_associated_hermite_to_hermite_diagonal_connection_coefficient)(norm1, norm2, n, c, D, 2);
    X2(create_associated_hermite_to_hermite_diagonal_connection_coefficient)(norm1, norm2, n, c, D+1, 2);
    X2(btb_eigen_FMM) * F2 = X2(btb_eig_FMM)(A, B, D);
    X(btb_eigen_FMM) * F = X(drop_precision_btb_eigen_FMM)(F2);
    X2(destroy_block_2x2_triangular_banded)(A);
    X2(destroy_block_2x2_triangular_banded)(B);
    X2(destroy_btb_eigen_FMM)(F2);
    free(D);
    return F;
}
