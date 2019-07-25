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

X(triangular_banded) * X(malloc_triangular_banded)(const int n, const int b) {
    FLT * data = (FLT *) malloc(n*(b+1)*sizeof(FLT));
    X(triangular_banded) * A = (X(triangular_banded) *) malloc(sizeof(X(triangular_banded)));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}

X(triangular_banded) * X(calloc_triangular_banded)(const int n, const int b) {
    FLT * data = (FLT *) calloc(n*(b+1), sizeof(FLT));
    X(triangular_banded) * A = (X(triangular_banded) *) malloc(sizeof(X(triangular_banded)));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}

FLT X(get_triangular_banded_index)(const X(triangular_banded) * A, const int i, const int j) {
    FLT * data = A->data;
    int n = A->n, b = A->b;
    if (0 <= j-i && j-i <= b && i < n && j < n)
        return data[i+(j+1)*b];
    else
        return 0;
}

void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j) {
    FLT * data = A->data;
    int n = A->n, b = A->b;
    if (0 <= j-i && j-i <= b && i < n && j < n)
        data[i+(j+1)*b] = v;
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
    X(tb_eigen_FMM) * F = (X(tb_eigen_FMM) *) malloc(sizeof(X(tb_eigen_FMM)));
    if (n < 64) {
        FLT * V = (FLT *) calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            V[i+i*n] = 1;
        F->lambda = (FLT *) malloc(n*sizeof(FLT));
        X(triangular_banded_eigenvalues)(A, B, F->lambda);
        X(triangular_banded_eigenvectors)(A, B, V);
        F->V = V;
        F->n = n;
        F->b = b;
    }
    else {
        F->lambda = (FLT *) malloc(n*sizeof(FLT));
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

        FLT * X = (FLT *) calloc(s*b, sizeof(FLT));
        for (int j = 0; j < b; j++) {
            X[s-b+j+j*s] = 1;
            X(tbsv)('N', B1, X+j*s);
            X(bfsv)('N', F->F1, X+j*s);
        }

        FLT * Y = (FLT *) calloc((n-s)*b, sizeof(FLT));
        for (int j = 0; j < b1; j++)
            for (int k = 0; k < b1-j; k++)
                Y[j+(k+j)*(n-s)] = A2->data[k+j*(b1+1)];
        FLT * Y2 = (FLT *) calloc((n-s)*b2, sizeof(FLT));
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
        F->t1 = (FLT *) calloc(s*b, sizeof(FLT));
        F->t2 = (FLT *) calloc((n-s)*b, sizeof(FLT));
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

// y ← A⁻¹*x, y ← A⁻ᵀ*x
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

// y ← A*x, y ← Aᵀ*x
void X(bfmv)(char TRANS, X(tb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    if (n < 64)
        X(trmv)(TRANS, n, F->V, x);
    else {
        int s = n/2, b = F->b;
        if (TRANS == 'N') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    F->t2[i+k*(n-s)] = F->Y[i+k*(n-s)]*x[i+s];
                X(himv)(TRANS, -1, F->F0, F->t2+k*(n-s), 0, F->t1+k*s);
                for (int i = 0; i < s; i++)
                    x[i] += F->t1[i+k*s]*F->X[i+k*s];
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
                    F->t1[i+k*s] = F->X[i+k*s]*x[i];
                X(himv)(TRANS, -1, F->F0, F->t1+k*s, 0, F->t2+k*(n-s));
                for (int i = 0; i < n-s; i++)
                    x[i+s] += F->t2[i+k*(n-s)]*F->Y[i+k*(n-s)];
            }
        }
    }
}

// y ← A⁻¹*x, y ← A⁻ᵀ*x
void X(bfsv)(char TRANS, X(tb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    if (n < 64)
        X(trsv)(TRANS, n, F->V, x);
    else {
        int s = n/2, b = F->b;
        if (TRANS == 'N') {
            X(bfsv)(TRANS, F->F1, x);
            X(bfsv)(TRANS, F->F2, x+s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    F->t2[i+k*(n-s)] = F->Y[i+k*(n-s)]*x[i+s];
                X(himv)(TRANS, 1, F->F0, F->t2+k*(n-s), 0, F->t1+k*s);
                for (int i = 0; i < s; i++)
                    x[i] += F->t1[i+k*s]*F->X[i+k*s];
            }
        }
        else if (TRANS == 'T') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    F->t1[i+k*s] = F->X[i+k*s]*x[i];
                X(himv)(TRANS, 1, F->F0, F->t1+k*s, 0, F->t2+k*(n-s));
                for (int i = 0; i < n-s; i++)
                    x[i+s] += F->t2[i+k*(n-s)]*F->Y[i+k*(n-s)];
            }
            X(bfsv)(TRANS, F->F1, x);
            X(bfsv)(TRANS, F->F2, x+s);
        }
    }
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
