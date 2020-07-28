static inline X2(triangular_banded) * X2(create_A_legendre_to_chebyshev)(const int n) {
    X2(triangular_banded) * A = X2(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X2(set_triangular_banded_index)(A, 2, 1, 1);
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(A, -i*(i-ONE(FLT2)), i-2, i);
        X2(set_triangular_banded_index)(A, i*(i+ONE(FLT2)), i, i);
    }
    return A;
}

static inline X2(triangular_banded) * X2(create_B_legendre_to_chebyshev)(const int n) {
    X2(triangular_banded) * B = X2(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X2(set_triangular_banded_index)(B, 2, 0, 0);
    if (n > 1)
        X2(set_triangular_banded_index)(B, 1, 1, 1);
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(B, -1, i-2, i);
        X2(set_triangular_banded_index)(B, 1, i, i);
    }
    return B;
}

X(tb_eigen_FMM) * X(plan_legendre_to_chebyshev)(const int normleg, const int normcheb, const int n) {
    X2(triangular_banded) * A = X2(create_A_legendre_to_chebyshev)(n);
    X2(triangular_banded) * B = X2(create_B_legendre_to_chebyshev)(n);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B);
    FLT2 * sclrow = malloc(n*sizeof(FLT2));
    FLT2 * sclcol = malloc(n*sizeof(FLT2));
    FLT2 t = 1, sqrtpi = Y2(tgamma)(0.5);
    FLT2 sqrtpi2 = sqrtpi/Y2(sqrt)(2);
    if (n > 0) {
        sclrow[0] = normcheb ? sqrtpi : 1;
        sclcol[0] = normleg ? Y2(sqrt)(0.5) : 1;
    }
    if (n > 1) {
        sclrow[1] = normcheb ? sqrtpi2 : 1;
        sclcol[1] = normleg ? Y2(sqrt)(1.5) : 1;
    }
    for (int i = 2; i < n; i++) {
        t *= (2*i-ONE(FLT2))/(2*i);
        sclrow[i] = normcheb ? sqrtpi2 : 1;
        sclcol[i] = (normleg ? Y2(sqrt)(i+0.5) : 1)*t;
    }
    X2(scale_rows_tb_eigen_FMM)(1, sclrow, F2);
    X2(scale_columns_tb_eigen_FMM)(1, sclcol, F2);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(sclrow);
    free(sclcol);
    return F;
}

static inline X2(triangular_banded) * X2(create_A_chebyshev_to_legendre)(const int n) {
    X2(triangular_banded) * A = X2(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X2(set_triangular_banded_index)(A, ONE(FLT2)/3, 1, 1);
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(A, -(i+1)/(2*i+ONE(FLT2))*(i+1), i-2, i);
        X2(set_triangular_banded_index)(A, i/(2*i+ONE(FLT2))*i, i, i);
    }
    return A;
}

static inline X2(triangular_banded) * X2(create_B_chebyshev_to_legendre)(const int n) {
    X2(triangular_banded) * B = X2(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X2(set_triangular_banded_index)(B, 1, 0, 0);
    if (n > 1)
        X2(set_triangular_banded_index)(B, ONE(FLT2)/3, 1, 1);
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(B, -1/(2*i+ONE(FLT2)), i-2, i);
        X2(set_triangular_banded_index)(B, 1/(2*i+ONE(FLT2)), i, i);
    }
    return B;
}

X(tb_eigen_FMM) * X(plan_chebyshev_to_legendre)(const int normcheb, const int normleg, const int n) {
    X2(triangular_banded) * A = X2(create_A_chebyshev_to_legendre)(n);
    X2(triangular_banded) * B = X2(create_B_chebyshev_to_legendre)(n);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B);
    FLT2 * sclrow = malloc(n*sizeof(FLT2));
    FLT2 * sclcol = malloc(n*sizeof(FLT2));
    FLT2 t = 1, sqrt_1_pi = 1/Y2(tgamma)(0.5);
    FLT2 sqrt_2_pi = Y2(sqrt)(2)*sqrt_1_pi;
    if (n > 0) {
        sclrow[0] = normleg ? 1/Y2(sqrt)(0.5) : 1;
        sclcol[0] = normcheb ? sqrt_1_pi : 1;
    }
    if (n > 1) {
        sclrow[1] = normleg ? 1/Y2(sqrt)(1.5) : 1;
        sclcol[1] = normcheb ? sqrt_2_pi : 1;
    }
    for (int i = 2; i < n; i++) {
        t *= (2*i)/(2*i-ONE(FLT2));
        sclrow[i] = normleg ? 1/Y2(sqrt)(i+0.5) : 1;
        sclcol[i] = (normcheb ? sqrt_2_pi : 1)*t;
    }
    X2(scale_rows_tb_eigen_FMM)(1, sclrow, F2);
    X2(scale_columns_tb_eigen_FMM)(1, sclcol, F2);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(sclrow);
    free(sclcol);
    return F;
}

static inline X2(triangular_banded) * X2(create_A_ultraspherical_to_ultraspherical)(const int n, const FLT2 lambda, const FLT2 mu) {
    X2(triangular_banded) * A = X2(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X2(set_triangular_banded_index)(A, (1+2*lambda)*mu/(1+mu), 1, 1);
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(A, -(i+2*mu)*(i+2*(mu-lambda))*mu/(i+mu), i-2, i);
        X2(set_triangular_banded_index)(A, i*(i+2*lambda)*mu/(i+mu), i, i);
    }
    return A;
}

static inline X2(triangular_banded) * X2(create_B_ultraspherical_to_ultraspherical)(const int n, const FLT2 mu) {
    X2(triangular_banded) * B = X2(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X2(set_triangular_banded_index)(B, 1, 0, 0);
    if (n > 1)
        X2(set_triangular_banded_index)(B, mu/(1+mu), 1, 1);
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(B, -mu/(i+mu), i-2, i);
        X2(set_triangular_banded_index)(B, mu/(i+mu), i, i);
    }
    return B;
}

X(tb_eigen_FMM) * X(plan_ultraspherical_to_ultraspherical)(const int norm1, const int norm2, const int n, const FLT lambda, const FLT mu) {
    X2(triangular_banded) * A = X2(create_A_ultraspherical_to_ultraspherical)(n, lambda, mu);
    X2(triangular_banded) * B = X2(create_B_ultraspherical_to_ultraspherical)(n, mu);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B);
    FLT2 * sclrow = malloc(n*sizeof(FLT2));
    FLT2 * sclcol = malloc(n*sizeof(FLT2));
    FLT2 lambda2 = lambda, mu2 = mu;
    if (n > 0) {
        sclrow[0] = norm2 ? Y2(sqrt)(Y2(tgamma)(0.5)*Y2(tgamma)(mu2+0.5)/Y2(tgamma)(mu2+1)) : 1;
        sclcol[0] = norm1 ? Y2(sqrt)(Y2(tgamma)(lambda2+1)/(Y2(tgamma)(0.5)*Y2(tgamma)(lambda2+0.5))) : 1;
    }
    for (int i = 1; i < n; i++) {
        sclrow[i] = norm2 ? Y2(sqrt)((i-1+mu2)/i*(i-1+2*mu2)/(i+mu2))*sclrow[i-1] : 1;
        sclcol[i] = norm1 ? Y2(sqrt)(i/(i-1+lambda2)*(i+lambda2)/(i-1+2*lambda2))*(i-1+lambda2)/(i-1+mu2)*sclcol[i-1] : (i-1+lambda2)/(i-1+mu2)*sclcol[i-1];
    }
    X2(scale_rows_tb_eigen_FMM)(1, sclrow, F2);
    X2(scale_columns_tb_eigen_FMM)(1, sclcol, F2);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(sclrow);
    free(sclcol);
    return F;
}

static inline X2(triangular_banded) * X2(create_A_jacobi_to_jacobi)(const int n, const FLT2 alpha, const FLT2 beta, const FLT2 gamma, const FLT2 delta) {
    X2(triangular_banded) * A = X2(malloc_triangular_banded)(n, 2);
    if (n > 0)
        X2(set_triangular_banded_index)(A, 0, 0, 0);
    if (n > 1) {
        X2(set_triangular_banded_index)(A, (gamma-delta)*(gamma+delta+2)/(gamma+delta+4)*(1+(gamma-alpha+delta-beta)/2) - (gamma+delta+2)*(gamma-alpha+beta-delta)/2, 0, 1);
        X2(set_triangular_banded_index)(A, (alpha+beta+2)*(gamma+delta+2)/(gamma+delta+4), 1, 1);
    }
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(A, -(i+gamma+delta+1)*(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1)*(i+gamma-alpha+delta-beta), i-2, i);
        X2(set_triangular_banded_index)(A, (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2)*(i*(i+gamma+delta+1)+(gamma+delta+2)*(gamma-alpha+delta-beta)/2) - (i+gamma+delta+1)*(gamma-alpha+beta-delta)/2, i-1, i);
        X2(set_triangular_banded_index)(A, i*(i+alpha+beta+1)*(i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2), i, i);
    }
    return A;
}

static inline X2(triangular_banded) * X2(create_B_jacobi_to_jacobi)(const int n, const FLT2 gamma, const FLT2 delta) {
    X2(triangular_banded) * B = X2(malloc_triangular_banded)(n, 2);
    if (n > 0)
        X2(set_triangular_banded_index)(B, 1, 0, 0);
    if (n > 1) {
        X2(set_triangular_banded_index)(B, (gamma-delta)/(gamma+delta+4), 0, 1);
        X2(set_triangular_banded_index)(B, (gamma+delta+2)/(gamma+delta+4), 1, 1);
    }
    for (int i = 2; i < n; i++) {
        X2(set_triangular_banded_index)(B, -(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1), i-2, i);
        X2(set_triangular_banded_index)(B, (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2), i-1, i);
        X2(set_triangular_banded_index)(B, (i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2), i, i);
    }
    return B;
}

X(tb_eigen_FMM) * X(plan_jacobi_to_jacobi)(const int norm1, const int norm2, const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X2(triangular_banded) * A = X2(create_A_jacobi_to_jacobi)(n, alpha, beta, gamma, delta);
    X2(triangular_banded) * B = X2(create_B_jacobi_to_jacobi)(n, gamma, delta);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B);
    FLT2 * sclrow = malloc(n*sizeof(FLT2));
    FLT2 * sclcol = malloc(n*sizeof(FLT2));
    FLT2 alpha2 = alpha, beta2 = beta, gamma2 = gamma, delta2 = delta;
    if (n > 0) {
        sclrow[0] = norm2 ? Y2(sqrt)(Y2(pow)(2, gamma2+delta2+1)*Y2(tgamma)(gamma2+1)*Y2(tgamma)(delta2+1)/Y2(tgamma)(gamma2+delta2+2)) : 1;
        sclcol[0] = norm1 ? Y2(sqrt)(Y2(tgamma)(alpha2+beta2+2)/(Y2(pow)(2, alpha2+beta2+1)*Y2(tgamma)(alpha2+1)*Y2(tgamma)(beta2+1))) : 1;
    }
    if (n > 1) {
        sclrow[1] = norm2 ? Y2(sqrt)((gamma2+1)*(delta2+1)/(gamma2+delta2+3))*sclrow[0] : 1;
        sclcol[1] = norm1 ? Y2(sqrt)((alpha2+beta2+3)/(alpha2+1)/(beta2+1))*(alpha2+beta2+2)/(gamma2+delta2+2)*sclcol[0] : (alpha2+beta2+2)/(gamma2+delta2+2);
    }
    for (int i = 2; i < n; i++) {
        sclrow[i] = norm2 ? Y2(sqrt)((i+gamma2)/i*(i+delta2)/(i+gamma2+delta2)*(2*i+gamma2+delta2-1)/(2*i+gamma2+delta2+1))*sclrow[i-1] : 1;
        sclcol[i] = norm1 ? Y2(sqrt)(i/(i+alpha2)*(i+alpha2+beta2)/(i+beta2)*(2*i+alpha2+beta2+1)/(2*i+alpha2+beta2-1))*(2*i+alpha2+beta2-1)/(i+alpha2+beta2)*(2*i+alpha2+beta2)/(2*i+gamma2+delta2-1)*(i+gamma2+delta2)/(2*i+gamma2+delta2)*sclcol[i-1] : (2*i+alpha2+beta2-1)/(i+alpha2+beta2)*(2*i+alpha2+beta2)/(2*i+gamma2+delta2-1)*(i+gamma2+delta2)/(2*i+gamma2+delta2)*sclcol[i-1];
    }
    X2(scale_rows_tb_eigen_FMM)(1, sclrow, F2);
    X2(scale_columns_tb_eigen_FMM)(1, sclcol, F2);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(sclrow);
    free(sclcol);
    return F;
}

static inline X2(triangular_banded) * X2(create_A_laguerre_to_laguerre)(const int n, const FLT2 alpha, const FLT2 beta) {
    X2(triangular_banded) * A = X2(malloc_triangular_banded)(n, 1);
    for (int i = 0; i < n; i++) {
        X2(set_triangular_banded_index)(A, alpha-beta-i, i-1, i);
        X2(set_triangular_banded_index)(A, i, i, i);
    }
    return A;
}

static inline X2(triangular_banded) * X2(create_B_laguerre_to_laguerre)(const int n) {
    X2(triangular_banded) * B = X2(malloc_triangular_banded)(n, 1);
    for (int i = 0; i < n; i++) {
        X2(set_triangular_banded_index)(B, -1, i-1, i);
        X2(set_triangular_banded_index)(B, 1, i, i);
    }
    return B;
}

X(tb_eigen_FMM) * X(plan_laguerre_to_laguerre)(const int norm1, const int norm2, const int n, const FLT alpha, const FLT beta) {
    X2(triangular_banded) * A = X2(create_A_laguerre_to_laguerre)(n, alpha, beta);
    X2(triangular_banded) * B = X2(create_B_laguerre_to_laguerre)(n);
    X2(tb_eigen_FMM) * F2 = X2(tb_eig_FMM)(A, B);
    FLT2 * sclrow = malloc(n*sizeof(FLT2));
    FLT2 * sclcol = malloc(n*sizeof(FLT2));
    FLT2 alpha2 = alpha, beta2 = beta;
    if (n > 0) {
        sclrow[0] = norm2 ? Y2(sqrt)(Y2(tgamma)(beta2+1)) : 1;
        sclcol[0] = norm1 ? 1/Y2(sqrt)(Y2(tgamma)(alpha2+1)) : 1;
    }
    for (int i = 1; i < n; i++) {
        sclrow[i] = norm2 ? Y2(sqrt)((i+beta2)/i)*sclrow[i-1] : 1;
        sclcol[i] = norm1 ? Y2(sqrt)(i/(i+alpha2))*sclcol[i-1] : 1;
    }
    X2(scale_rows_tb_eigen_FMM)(1, sclrow, F2);
    X2(scale_columns_tb_eigen_FMM)(1, sclcol, F2);
    X(tb_eigen_FMM) * F = X(drop_precision_tb_eigen_FMM)(F2);
    X2(destroy_triangular_banded)(A);
    X2(destroy_triangular_banded)(B);
    X2(destroy_tb_eigen_FMM)(F2);
    free(sclrow);
    free(sclcol);
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
