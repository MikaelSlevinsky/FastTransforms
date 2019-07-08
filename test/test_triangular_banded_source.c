X(triangular_banded) * X(eigenplan_A_jac2jac)(const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X(triangular_banded) * A = X(malloc_triangular_banded)(n, 2);
    FLT v;
    if (n > 0)
        X(set_triangular_banded_index)(A, 0, 0, 0);
    for (int i = 1; i < n; i++) {
        v = i*(i+alpha+beta+1)*(i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2);
        X(set_triangular_banded_index)(A, v, i, i);
    }
    for (int i = 1; i < n; i++) {
        v = (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2)*(i*(i+gamma+delta+1)+(gamma+delta+2)*(gamma+delta-alpha-beta)/2) - (i+gamma+delta+1)*(gamma-alpha+beta-delta)/2;
        X(set_triangular_banded_index)(A, v, i-1, i);
    }
    for (int i = 2; i < n; i++) {
        v = -(i+gamma+delta+1)*(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1)*(i+gamma-alpha+delta-beta);
        X(set_triangular_banded_index)(A, v, i-2, i);
    }
    return A;
}

X(triangular_banded) * X(eigenplan_B_jac2jac)(const int n, const FLT gamma, const FLT delta) {
    X(triangular_banded) * B = X(malloc_triangular_banded)(n, 2);
    FLT v;
    if (n > 0)
        X(set_triangular_banded_index)(B, 1, 0, 0);
    for (int i = 1; i < n; i++) {
        v = (i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2);
        X(set_triangular_banded_index)(B, v, i, i);
    }
    for (int i = 1; i < n; i++) {
        v = (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2);
        X(set_triangular_banded_index)(B, v, i-1, i);
    }
    for (int i = 2; i < n; i++) {
        v = -(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1);
        X(set_triangular_banded_index)(B, v, i-2, i);
    }
    return B;
}


void Y(test_triangular_banded)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 20;
    int b = 2;
    //FLT alpha = -0.25, beta = 0.25, gamma = 0.25, delta = 0.0;
    //FLT alpha = -0.5, beta = 0.5, gamma = 0.0, delta = 0.0;
    //FLT alpha = -0.5, beta = 0.5, gamma = 1.5, delta = 2.5;
    FLT alpha = 0.0, beta = -0.5, gamma = -0.5, delta = -0.5;

    X(triangular_banded) * A = X(eigenplan_A_jac2jac)(n, alpha, beta, gamma, delta);
    X(triangular_banded) * B = X(eigenplan_B_jac2jac)(n, gamma, delta);
    /*
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
    */

    FLT * V = (FLT *) calloc(n*n, sizeof(FLT));
    //for (int i = 0; i < n; i++)
    //    V[i+i*n] = 1.0;
    V[0] = 1.0;
    V[1+n] = (alpha+beta+2)/(gamma+delta+2);
    for (int i = 2; i < n; i++) {
        V[i+i*n] = (2*i+alpha+beta-1)/(i+alpha+beta)*(2*i+alpha+beta)/(2*i+gamma+delta-1)*(i+gamma+delta)/(2*i+gamma+delta)*V[i-1+(i-1)*n];
    }
    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambda[i] = i*(i+alpha+beta+1);

    //X(triangular_banded_eigenvalues)(A, B, lambda);
    printmat("Λ", "%3.3f", lambda, n, 1);
    X(triangular_banded_eigenvectors)(A, B, V);
    printmat("V", "%1.3f", V, n, n);

    printf("This is A[0,0]: %1.3f", X(get_triangular_banded_index)(A, 0, 0));
    printf("This is B[0,0]: %1.3f", X(get_triangular_banded_index)(B, 0, 0));

    X(destroy_triangular_banded)(A);
    X(destroy_triangular_banded)(B);
    free(V);
    free(lambda);
}
