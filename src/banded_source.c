void X(destroy_banded)(X(banded) * A) {
    free(A->data);
    free(A);
}

void X(destroy_triangular_banded)(X(triangular_banded) * A) {
    free(A->data);
    free(A);
}

void X(destroy_tb_eigen_FMM)(X(tb_eigen_FMM) * F) {
    if (F->n < 64) {
        free(F->V);
        free(F->lambda);
    }
    else {
        X(destroy_hierarchicalmatrix)(F->F0);
        X(destroy_tb_eigen_FMM)(F->F1);
        X(destroy_tb_eigen_FMM)(F->F2);
        free(F->X);
        free(F->Y);
        free(F->t1);
        free(F->t2);
        free(F->lambda);
    }
    free(F);
}

size_t X(summary_size_tb_eigen_FMM)(X(tb_eigen_FMM) * F) {
    size_t S = 0;
    if (F->n < 64)
        S += sizeof(FLT)*F->n*(F->n+1);
    else {
        S += X(summary_size_hierarchicalmatrix)(F->F0);
        S += X(summary_size_tb_eigen_FMM)(F->F1);
        S += X(summary_size_tb_eigen_FMM)(F->F2);
        S += sizeof(FLT)*F->n*(2*F->b+1);
    }
    return S;
}

X(banded) * X(malloc_banded)(const int m, const int n, const int l, const int u) {
    FLT * data = malloc(n*(l+u+1)*sizeof(FLT));
    X(banded) * A = malloc(sizeof(X(banded)));
    A->data = data;
    A->m = m;
    A->n = n;
    A->l = l;
    A->u = u;
    return A;
}
X(banded) * X(calloc_banded)(const int m, const int n, const int l, const int u) {
    FLT * data = calloc(n*(l+u+1), sizeof(FLT));
    X(banded) * A = malloc(sizeof(X(banded)));
    A->data = data;
    A->m = m;
    A->n = n;
    A->l = l;
    A->u = u;
    return A;
}

X(triangular_banded) * X(malloc_triangular_banded)(const int n, const int b) {
    FLT * data = malloc(n*(b+1)*sizeof(FLT));
    X(triangular_banded) * A = malloc(sizeof(X(triangular_banded)));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}

X(triangular_banded) * X(calloc_triangular_banded)(const int n, const int b) {
    FLT * data = calloc(n*(b+1), sizeof(FLT));
    X(triangular_banded) * A = malloc(sizeof(X(triangular_banded)));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}

FLT X(get_banded_index)(const X(banded) * A, const int i, const int j) {
    FLT * data = A->data;
    int m = A->m, n = A->n, l = A->l, u = A->u;
    if (0 <= i && 0 <= j && -l <= j-i && j-i <= u && i < m && j < n)
        return data[u+i-j+j*(l+u+1)];
    else
        return 0;
}

void X(set_banded_index)(const X(banded) * A, const FLT v, const int i, const int j) {
    FLT * data = A->data;
    int m = A->m, n = A->n, l = A->l, u = A->u;
    if (0 <= i && 0 <= j && -l <= j-i && j-i <= u && i < m && j < n)
        data[u+i-j+j*(l+u+1)] = v;
}

FLT X(get_triangular_banded_index)(const X(triangular_banded) * A, const int i, const int j) {
    FLT * data = A->data;
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j-i && j-i <= b && i < n && j < n)
        return data[i+(j+1)*b];
    else
        return 0;
}

void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j) {
    FLT * data = A->data;
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j-i && j-i <= b && i < n && j < n)
        data[i+(j+1)*b] = v;
}

// y ← α*A*x + β*y
void X(gbmv)(FLT alpha, X(banded) * A, FLT * x, FLT beta, FLT * y) {
    FLT ab, c;
    int m = A->m, n = A->n, l = A->l, u = A->u;
    for (int i = 0; i < m; i++)
        y[i] = beta*y[i];
    for (int i = 0; i < m; i++)
        for (int j = MAX(0, i-l); j < MIN(n, i+u+1); j++)
            y[i] += X(get_banded_index)(A, i, j)*x[j];
}

// C ← α*A*B + β*C
void X(gbmm)(FLT alpha, X(banded) * A, X(banded) * B, FLT beta, X(banded) * C) {
    FLT ab, c;
    int m = A->m, n = A->n, p = B->n;
    int l = C->l, u = C->u, l1 = A->l, u1 = A->u, l2 = B->l, u2 = B->u;
    if (C->m != m || B->m != n || C->n != p)
        printf("GBMM: Sizes are off.\n");
    if (C->l < l1+l2 || C->u < u1+u2)
        printf("GBMM: Bandwidths are off.\n");
    for (int j = 0; j < p; j++)
        for (int i = MAX(0, j-u); i < MIN(m, j+l+1); i++) {
            ab = 0;
            for (int k = MAX(MAX(0, i-l1), j-u2); k < MIN(MIN(n, i+u1+1), j+l2+1); k++)
                ab += X(get_banded_index)(A, i, k)*X(get_banded_index)(B, k, j);
            c = X(get_banded_index)(C, i, j);
            X(set_banded_index)(C, alpha*ab+beta*c, i, j);
        }
}

// C ← α*A+β*B
void X(banded_add)(FLT alpha, X(banded) * A, FLT beta, X(banded) * B, X(banded) * C) {
    int m = C->m, n = C->n, l = C->l, u = C->u;
    if (A->m != m || B->m != m || A->n != n || B->n != n)
        printf("GBMM: Sizes are off.\n");
    if (l < MAX(A->l, B->l) || C->u < MAX(A->u, B->u))
        printf("GBMM: Bandwidths are off.\n");
    for (int j = 0; j < n; j++)
        for (int i = MAX(0, j-u); i < MIN(m, j+l+1); i++)
            X(set_banded_index)(C, alpha*X(get_banded_index)(A, i, j) + beta*X(get_banded_index)(B, i, j), i, j);
}

// x ← A*x, x ← Aᵀ*x
void X(tbmv)(char TRANS, X(triangular_banded) * A, FLT * x) {
    int n = A->n, b = A->b;
    FLT * data = A->data, t;
    if (TRANS == 'N') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = i; k < MIN(i+b+1, n); k++)
                t += data[i+(k+1)*b]*x[k];
            x[i] = t;
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = MAX(i-b, 0); k <= i; k++)
                t += data[k+(i+1)*b]*x[k];
            x[i] = t;
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(tbsv)(char TRANS, X(triangular_banded) * A, FLT * x) {
    int n = A->n, b = A->b;
    FLT * data = A->data, t;
    if (TRANS == 'N') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = i+1; k < MIN(i+b+1, n); k++)
                t += data[i+(k+1)*b]*x[k];
            x[i] = (x[i] - t)/data[i+(i+1)*b];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = MAX(i-b, 0); k < i; k++)
                t += data[k+(i+1)*b]*x[k];
            x[i] = (x[i] - t)/data[i+(i+1)*b];
        }
    }
}

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda) {
    for (int j = 0; j < A->n; j++)
        lambda[j] = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
}

// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    FLT t, lam;
    for (int j = 1; j < n; j++) {
        lam = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
        for (int i = j-1; i >= 0; i--) {
            t = 0;
            for (int k = i+1; k < MIN(i+b+1, n); k++)
                t += (X(get_triangular_banded_index)(A, i, k) - lam*X(get_triangular_banded_index)(B, i, k))*V[k+j*n];
            V[i+j*n] = t/(lam*X(get_triangular_banded_index)(B, i, i) - X(get_triangular_banded_index)(A, i, i));
        }
    }
}

X(tb_eigen_FMM) * X(tb_eig_FMM)(X(triangular_banded) * A, X(triangular_banded) * B) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    X(tb_eigen_FMM) * F = malloc(sizeof(X(tb_eigen_FMM)));
    if (n < 64) {
        FLT * V = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            V[i+i*n] = 1;
        F->lambda = malloc(n*sizeof(FLT));
        X(triangular_banded_eigenvalues)(A, B, F->lambda);
        X(triangular_banded_eigenvectors)(A, B, V);
        F->V = V;
        F->n = n;
        F->b = b;
    }
    else {
        F->lambda = malloc(n*sizeof(FLT));
        X(triangular_banded_eigenvalues)(A, B, F->lambda);
        int s = n/2;
        X(triangular_banded) * A1 = X(calloc_triangular_banded)(s, b1);
        X(triangular_banded) * B1 = X(calloc_triangular_banded)(s, b2);
        for (int j = 0; j < s; j++)
            for (int k = 0; k < b1+1; k++)
                A1->data[k+j*(b1+1)] = A->data[k+j*(b1+1)];
        for (int j = 0; j < s; j++)
            for (int k = 0; k < b2+1; k++)
                B1->data[k+j*(b2+1)] = B->data[k+j*(b2+1)];
        A1->n = B1->n = s;
        A1->b = b1;
        B1->b = b2;

        X(triangular_banded) * A2 = X(calloc_triangular_banded)(n-s, b1);
        X(triangular_banded) * B2 = X(calloc_triangular_banded)(n-s, b2);
        for (int j = 0; j < n-s; j++)
            for (int k = 0; k < b1+1; k++)
                A2->data[k+j*(b1+1)] = A->data[k+(j+s)*(b1+1)];
        for (int j = 0; j < n-s; j++)
            for (int k = 0; k < b2+1; k++)
                B2->data[k+j*(b2+1)] = B->data[k+(j+s)*(b2+1)];
        A2->n = B2->n = n-s;
        A2->b = b1;
        B2->b = b2;

        F->F1 = X(tb_eig_FMM)(A1, B1);
        F->F2 = X(tb_eig_FMM)(A2, B2);

        FLT * lambda1 = F->F1->lambda;
        FLT * lambda2 = F->F2->lambda;

        FLT * X = calloc(s*b, sizeof(FLT));
        for (int j = 0; j < b; j++) {
            X[s-b+j+j*s] = 1;
            X(tbsv)('N', B1, X+j*s);
            X(bfsv)('N', F->F1, X+j*s);
        }

        FLT * Y = calloc((n-s)*b, sizeof(FLT));
        for (int j = 0; j < b1; j++)
            for (int k = 0; k < b1-j; k++)
                Y[j+(k+j)*(n-s)] = A2->data[k+j*(b1+1)];
        FLT * Y2 = calloc((n-s)*b2, sizeof(FLT));
        for (int j = 0; j < b2; j++)
            for (int k = 0; k < b2-j; k++)
                Y2[j+(k+j)*(n-s)] = B2->data[k+j*(b2+1)];

        for (int j = 0; j < b1; j++)
            X(bfmv)('T', F->F2, Y+j*(n-s));
        for (int j = 0; j < b2; j++)
            X(bfmv)('T', F->F2, Y2+j*(n-s));
        for (int j = 0; j < b2; j++)
            for (int i = 0; i < n-s; i++)
                Y2[i+j*(n-s)] *= lambda2[i];
        for (int j = 0; j < b2; j++)
            for (int i = 0; i < n-s; i++)
                Y[i+j*(n-s)] = Y[i+j*(n-s)]-Y2[i+j*(n-s)];

        F->F0 = X(sample_hierarchicalmatrix)(X(cauchykernel), lambda1, lambda2, (unitrange) {0, s}, (unitrange) {0, n-s});
        F->X = X;
        F->Y = Y;
        F->t1 = calloc(s*FT_GET_MAX_THREADS(), sizeof(FLT));
        F->t2 = calloc((n-s)*FT_GET_MAX_THREADS(), sizeof(FLT));
        F->n = n;
        F->b = b;
        X(destroy_triangular_banded)(A1);
        X(destroy_triangular_banded)(B1);
        X(destroy_triangular_banded)(A2);
        X(destroy_triangular_banded)(B2);
        free(Y2);
    }
    return F;
}

void X(scale_rows_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F) {
    int n = F->n;
    if (n < 64) {
        FLT * V = F->V;
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= alpha*x[i];
    }
    else {
        int s = n/2;
        X(scale_rows_tb_eigen_FMM)(alpha, x, F->F1);
        X(scale_rows_tb_eigen_FMM)(alpha, x+s, F->F2);
    }
}

void X(scale_columns_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F) {
    int n = F->n;
    if (n < 64) {
        FLT scl, * V = F->V;
        for (int j = 0; j < n; j++) {
            scl = alpha*x[j];
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= scl;
        }
    }
    else {
        int s = n/2, b = F->b;
        for (int k = 0; k < b; k++) {
            for (int i = 0; i < s; i++)
                F->X[i+k*s] /= x[i];
            for (int i = 0; i < n-s; i++)
                F->Y[i+k*(n-s)] *= x[i+s];
        }
        X(scale_columns_tb_eigen_FMM)(alpha, x, F->F1);
        X(scale_columns_tb_eigen_FMM)(alpha, x+s, F->F2);
    }
}


// y ← A*x, y ← Aᵀ*x
void X(trmv)(char TRANS, int n, FLT * A, FLT * x) {
    if (TRANS == 'N') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++)
                x[i] += A[i+j*n]*x[j];
            x[j] *= A[j+j*n];
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            x[i] *= A[i+i*n];
            for (int j = i-1; j >= 0; j--)
                x[i] += A[j+i*n]*x[j];
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(trsv)(char TRANS, int n, FLT * A, FLT * x) {
    if (TRANS == 'N') {
        for (int j = n-1; j >= 0; j--) {
            x[j] /= A[j+j*n];
            for (int i = 0; i < j; i++)
                x[i] -= A[i+j*n]*x[j];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++)
                x[i] -= A[j+i*n]*x[j];
            x[i] /= A[i+i*n];
        }
    }
}

void X(trmm)(char TRANS, int n, FLT * A, FLT * X, int LDX, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(trmv)(TRANS, n, A, X+j*LDX);
}

void X(trsm)(char TRANS, int n, FLT * A, FLT * X, int LDX, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(trsv)(TRANS, n, A, X+j*LDX);
}

// x ← A*x, x ← Aᵀ*x
void X(bfmv)(char TRANS, X(tb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    if (n < 64)
        X(trmv)(TRANS, n, F->V, x);
    else {
        int s = n/2, b = F->b;
        FLT * t1 = F->t1+s*FT_GET_THREAD_NUM(), * t2 = F->t2+(n-s)*FT_GET_THREAD_NUM();
        if (TRANS == 'N') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    t2[i] = F->Y[i+k*(n-s)]*x[i+s];
                X(ghmv)(TRANS, -1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[i] += t1[i]*F->X[i+k*s];
            }
            X(bfmv)(TRANS, F->F1, x);
            X(bfmv)(TRANS, F->F2, x+s);
        }
        else if (TRANS == 'T') {
            X(bfmv)(TRANS, F->F1, x);
            X(bfmv)(TRANS, F->F2, x+s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    t1[i] = F->X[i+k*s]*x[i];
                X(ghmv)(TRANS, -1, F->F0, t1, 0, t2);
                for (int i = 0; i < n-s; i++)
                    x[i+s] += t2[i]*F->Y[i+k*(n-s)];
            }
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(bfsv)(char TRANS, X(tb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    if (n < 64)
        X(trsv)(TRANS, n, F->V, x);
    else {
        int s = n/2, b = F->b;
        FLT * t1 = F->t1+s*FT_GET_THREAD_NUM(), * t2 = F->t2+(n-s)*FT_GET_THREAD_NUM();
        if (TRANS == 'N') {
            X(bfsv)(TRANS, F->F1, x);
            X(bfsv)(TRANS, F->F2, x+s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    t2[i] = F->Y[i+k*(n-s)]*x[i+s];
                X(ghmv)(TRANS, 1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[i] += t1[i]*F->X[i+k*s];
            }
        }
        else if (TRANS == 'T') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    t1[i] = F->X[i+k*s]*x[i];
                X(ghmv)(TRANS, 1, F->F0, t1, 0, t2);
                for (int i = 0; i < n-s; i++)
                    x[i+s] += t2[i]*F->Y[i+k*(n-s)];
            }
            X(bfsv)(TRANS, F->F1, x);
            X(bfsv)(TRANS, F->F2, x+s);
        }
    }
}

void X(bfmm)(char TRANS, X(tb_eigen_FMM) * F, FLT * X, int LDX, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bfmv)(TRANS, F, X+j*LDX);
}

void X(bfsm)(char TRANS, X(tb_eigen_FMM) * F, FLT * X, int LDX, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bfsv)(TRANS, F, X+j*LDX);
}

X(triangular_banded) * X(create_A_legendre_to_chebyshev)(const int n) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X(set_triangular_banded_index)(A, 2, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, -i*(i-ONE(FLT)), i-2, i);
        X(set_triangular_banded_index)(A, i*(i+ONE(FLT)), i, i);
    }
    return A;
}

X(triangular_banded) * X(create_B_legendre_to_chebyshev)(const int n) {
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

X(triangular_banded) * X(create_A_chebyshev_to_legendre)(const int n) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X(set_triangular_banded_index)(A, ONE(FLT)/3, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, -(i+1)/(2*i+ONE(FLT))*(i+1), i-2, i);
        X(set_triangular_banded_index)(A, i/(2*i+ONE(FLT))*i, i, i);
    }
    return A;
}

X(triangular_banded) * X(create_B_chebyshev_to_legendre)(const int n) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(B, 1, 0, 0);
    if (n > 1)
        X(set_triangular_banded_index)(B, ONE(FLT)/3, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(B, -1/(2*i+ONE(FLT)), i-2, i);
        X(set_triangular_banded_index)(B, 1/(2*i+ONE(FLT)), i, i);
    }
    return B;
}

X(triangular_banded) * X(create_A_ultraspherical_to_ultraspherical)(const int n, const FLT lambda, const FLT mu) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X(set_triangular_banded_index)(A, (1+2*lambda)*mu/(1+mu), 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, -(i+2*mu)*(i+2*(mu-lambda))*mu/(i+mu), i-2, i);
        X(set_triangular_banded_index)(A, i*(i+2*lambda)*mu/(i+mu), i, i);
    }
    return A;
}

X(triangular_banded) * X(create_B_ultraspherical_to_ultraspherical)(const int n, const FLT mu) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(B, 1, 0, 0);
    if (n > 1)
        X(set_triangular_banded_index)(B, mu/(1+mu), 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(B, -mu/(i+mu), i-2, i);
        X(set_triangular_banded_index)(B, mu/(i+mu), i, i);
    }
    return B;
}

X(triangular_banded) * X(create_A_jacobi_to_jacobi)(const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X(triangular_banded) * A = X(malloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(A, 0, 0, 0);
    if (n > 1) {
        X(set_triangular_banded_index)(A, (gamma-delta)*(gamma+delta+2)/(gamma+delta+4)*(1+(gamma-alpha+delta-beta)/2) - (gamma+delta+2)*(gamma-alpha+beta-delta)/2, 0, 1);
        X(set_triangular_banded_index)(A, (alpha+beta+2)*(gamma+delta+2)/(gamma+delta+3)*(gamma+delta+3)/(gamma+delta+4), 1, 1);
    }
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, -(i+gamma+delta+1)*(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1)*(i+gamma-alpha+delta-beta), i-2, i);
        X(set_triangular_banded_index)(A, (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2)*(i*(i+gamma+delta+1)+(gamma+delta+2)*(gamma-alpha+delta-beta)/2) - (i+gamma+delta+1)*(gamma-alpha+beta-delta)/2, i-1, i);
        X(set_triangular_banded_index)(A, i*(i+alpha+beta+1)*(i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2), i, i);
    }
    return A;
}

X(triangular_banded) * X(create_B_jacobi_to_jacobi)(const int n, const FLT gamma, const FLT delta) {
    X(triangular_banded) * B = X(malloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(B, 1, 0, 0);
    if (n > 1) {
        X(set_triangular_banded_index)(B, (gamma-delta)/(gamma+delta+4), 0, 1);
        X(set_triangular_banded_index)(B, (gamma+delta+2)/(gamma+delta+4), 1, 1);
    }
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(B, -(i+gamma)/(2*i+gamma+delta)*(i+delta)/(2*i+gamma+delta+1), i-2, i);
        X(set_triangular_banded_index)(B, (gamma-delta)*(i+gamma+delta+1)/(2*i+gamma+delta)/(2*i+gamma+delta+2), i-1, i);
        X(set_triangular_banded_index)(B, (i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2), i, i);
    }
    return B;
}

#define delta(k) (((k)%2) ? 1 : 0)

X(triangular_banded) * X(create_A_konoplev_to_jacobi)(const int n, const FLT alpha, const FLT beta) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(A, 0, 0, 0);
    if (n > 1) {
        X(set_triangular_banded_index)(A, 3*(2*alpha+2*beta+3)/(2*alpha+5), 1, 1);
    }
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, (i-2*beta-1)*(i+2*alpha+1)/(2*i+2*alpha-1)*(i+alpha-1)/(2*i+2*alpha+1)*(i+alpha), i-2, i);
        X(set_triangular_banded_index)(A, i*(i+2*alpha+2*beta+2)*(i+1)/(i+2-delta(i))*(i+2)/(i+2*alpha+2-delta(i))*(i+2*alpha+1)/(2*i+2*alpha+1)*(i+2*alpha+2)/(2*i+2*alpha+3), i, i);
    }
    return A;
}

X(triangular_banded) * X(create_B_konoplev_to_jacobi)(const int n, const FLT alpha) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(B, 1/(2*alpha+3), 0, 0);
    if (n > 1) {
        X(set_triangular_banded_index)(B, 3/(2*alpha+5), 1, 1);
    }
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(B, (i+alpha-1)/(2*i+2*alpha-1)*(i+alpha)/(2*i+2*alpha+1), i-2, i);
        X(set_triangular_banded_index)(B, (i+1+delta(i))/(2*i+2*alpha+1)*(i+2*alpha+1+delta(i))/(2*i+2*alpha+3), i, i);
    }
    return B;
}

#undef delta

X(banded) * X(create_jacobi_derivative)(const int m, const int n, const int order, const FLT alpha, const FLT beta) {
    X(banded) * A = X(malloc_banded)(m, n, -order, order);
    FLT v;
    for (int j = order; j < n; j++) {
        v = 1;
        for (int k = 0; k < order; k++)
            v *= (j+alpha+beta+k+1)/2;
        X(set_banded_index)(A, v, j-order, j);
    }
    return A;
}

// x P^{(α,β)}
X(banded) * X(create_jacobi_multiplication)(const int m, const int n, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(m, n, 1, 1);
    FLT v;
    for (int j = 0; j < n; j++) {
        v = 2*(j+alpha)/(2*j+alpha+beta)*(j+beta)/(2*j+alpha+beta+1);
        X(set_banded_index)(A, v, j-1, j);
        if (j == 0)
            v = (beta-alpha)/(alpha+beta+2);
        else
            v = (beta-alpha)*(alpha+beta)/(2*j+alpha+beta)/(2*j+alpha+beta+2);
        X(set_banded_index)(A, v, j, j);
        v = 2*(j+1)/(2*j+alpha+beta+1)*(j+alpha+beta+1)/(2*j+alpha+beta+2);
        X(set_banded_index)(A, v, j+1, j);
    }
    return A;
}

// P^{(α,β)} ↗ P^{(α+1,β+1)}
X(banded) * X(create_jacobi_raising)(const int m, const int n, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(m, n, 0, 2);
    FLT v;
    for (int j = 0; j < n; j++) {
        v = -(j+alpha)/(2*j+alpha+beta)*(j+beta)/(2*j+alpha+beta+1);
        X(set_banded_index)(A, v, j-2, j);
        v = (alpha-beta)/(2*j+alpha+beta)*(j+alpha+beta+1)/(2*j+alpha+beta+2);
        X(set_banded_index)(A, v, j-1, j);
        v = (j+alpha+beta+1)/(2*j+alpha+beta+1)*(j+alpha+beta+2)/(2*j+alpha+beta+2);
        X(set_banded_index)(A, v, j, j);
    }
    return A;
}

// (1-x²) P^{(α+1,β+1)} ↘ P^{(α,β)}
X(banded) * X(create_jacobi_lowering)(const int m, const int n, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(m, n, 2, 0);
    FLT v;
    for (int j = 0; j < n; j++) {
        v = 4*(j+alpha+1)/(2*j+alpha+beta+2)*(j+beta+1)/(2*j+alpha+beta+3);
        X(set_banded_index)(A, v, j, j);
        v = 4*(alpha-beta)/(2*j+alpha+beta+2)*(j+1)/(2*j+alpha+beta+4);
        X(set_banded_index)(A, v, j+1, j);
        v = -4*(j+1)/(2*j+alpha+beta+3)*(j+2)/(2*j+alpha+beta+4);
        X(set_banded_index)(A, v, j+2, j);
    }
    return A;
}

X(triangular_banded) * X(create_A_associated_jacobi_to_jacobi)(const int n, const int c, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X(banded) * A = X(calloc_banded)(n, n, 0, 4);

    X(banded) * D1 = X(create_jacobi_derivative)(n, n, 1, gamma, delta);
    X(banded) * D2 = X(create_jacobi_derivative)(n, n, 2, gamma, delta);
    X(banded) * D3 = X(create_jacobi_derivative)(n, n, 3, gamma, delta);
    X(banded) * D4 = X(create_jacobi_derivative)(n, n, 4, gamma, delta);
    X(banded) * R1 = X(create_jacobi_raising)(n, n, gamma+1, delta+1);
    X(banded) * L1 = X(create_jacobi_lowering)(n, n, gamma+1, delta+1);
    X(banded) * L2 = X(create_jacobi_lowering)(n, n, gamma+2, delta+2);
    X(banded) * L3 = X(create_jacobi_lowering)(n, n, gamma+3, delta+3);
    X(banded) * M1 = X(create_jacobi_multiplication)(n, n, gamma+1, delta+1);
    X(banded) * M2 = X(create_jacobi_multiplication)(n, n, gamma+2, delta+2);

    // A4 = (1-x²)² D⁴

    X(banded) * A4a = X(calloc_banded)(n, n, -2, 4);
    X(gbmm)(1, L3, D4, 0, A4a);
    X(banded) * A4 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, L2, A4a, 0, A4);

    // A3 = -10 x (1-x²) D³

    X(banded) * A3a = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, L2, D3, 0, A3a);
    X(banded) * A3 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(-10, M2, A3a, 0, A3);

    // A2 = [ (μ+(n+1)²-24)(1-x²)-2ν(1+x)-4β²+16 ] D²
    //    = [ (μ+(n+1)²-24)(1-x²)-2νx -2ν-4β²+16 ] D²

    X(banded) * A2 = X(calloc_banded)(n, n, 0, 4);
    X(banded) * A2a = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(1, L1, D2, 0, A2a);
    X(banded) * A2b = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, R1, A2a, 0, A2b);
    X(banded) * A2c = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, M2, D2, 0, A2c);
    FLT * D2b = calloc(n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        D2b[j] = (j+alpha+beta+2*c-1)*(j+alpha+beta+2*c+1)+(j+1)*(j+ONE(FLT))-24;
        for (int k = 0; k < 5; k++)
            A2b->data[k+j*5] *= D2b[j];
    }
    X(banded_add)(1, A2b, -2*(alpha-beta)*(alpha+beta), A2c, A2);
    X(banded_add)(1, A2, -2*(alpha-beta)*(alpha+beta)-4*beta*beta+16, D2, A2);

    // A1 = [ -3*(μ+(n+2)(n-1))x - ν ] D

    X(banded) * A1 = X(calloc_banded)(n, n, 0, 4);
    X(banded) * A1a = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(1, M1, D1, 0, A1a);
    X(banded) * A1b = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, R1, A1a, 0, A1b);
    X(banded) * A1c = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, R1, D1, 0, A1c);
    FLT * D1b = calloc(n, sizeof(FLT));
    for (int j = 0; j < n; j++) {
        D1b[j] = (j+alpha+beta+2*c-1)*(j+alpha+beta+2*c+1) + (j+3)*(j-ONE(FLT));
        for (int k = 0; k < 5; k++)
            A1b->data[k+j*5] *= D1b[j];
    }
    X(banded_add)(-3, A1b, -(alpha-beta)*(alpha+beta), A1c, A1);

    // A = -(A1+A2+A3+A4)

    X(banded_add)(-1, A1, -1, A2, A);
    X(banded_add)(1, A, -1, A3, A);
    X(banded_add)(1, A, -1, A4, A);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(D3);
    X(destroy_banded)(D4);
    X(destroy_banded)(R1);
    X(destroy_banded)(L1);
    X(destroy_banded)(L2);
    X(destroy_banded)(L3);
    X(destroy_banded)(M1);
    X(destroy_banded)(M2);

    X(destroy_banded)(A4a);
    X(destroy_banded)(A4);
    X(destroy_banded)(A3a);
    X(destroy_banded)(A3);
    X(destroy_banded)(A2a);
    X(destroy_banded)(A2b);
    X(destroy_banded)(A2c);
    X(destroy_banded)(A2);
    X(destroy_banded)(A1a);
    X(destroy_banded)(A1b);
    X(destroy_banded)(A1c);
    X(destroy_banded)(A1);
    free(D2b);
    free(D1b);

    X(triangular_banded) * TA = malloc(sizeof(X(triangular_banded)));
    TA->data = A->data;
    TA->n = n;
    TA->b = 4;
    free(A);
    return TA;
}

X(triangular_banded) * X(create_B_associated_jacobi_to_jacobi)(const int n, const FLT gamma, const FLT delta) {
    X(banded) * B = X(calloc_banded)(n, n, 0, 4);

    X(banded) * R0 = X(create_jacobi_raising)(n, n, gamma, delta);
    X(banded) * R1 = X(create_jacobi_raising)(n, n, gamma+1, delta+1);

    X(gbmm)(1, R1, R0, 0, B);

    X(destroy_banded)(R0);
    X(destroy_banded)(R1);

    X(triangular_banded) * TB = malloc(sizeof(X(triangular_banded)));
    TB->data = B->data;
    TB->n = n;
    TB->b = 4;
    free(B);
    return TB;
}
