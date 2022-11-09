void X(destroy_sparse)(X(sparse) * A) {
    free(A->p);
    free(A->q);
    free(A->v);
    free(A);
}

void X(destroy_banded)(X(banded) * A) {
    free(A->data);
    free(A);
}

void X(destroy_triangular_banded)(X(triangular_banded) * A) {
    free(A->data);
    free(A);
}

static inline void X(destroy_banded_orthogonal_triangular)(struct X(banded_orthogonal_triangular) * F) {
    X(destroy_banded)(F->factors);
    free(F->tau);
    free(F);
}

void X(destroy_banded_qr)(X(banded_qr) * F) {X(destroy_banded_orthogonal_triangular)(F);}
void X(destroy_banded_ql)(X(banded_ql) * F) {X(destroy_banded_orthogonal_triangular)(F);}

void X(destroy_tb_eigen_FMM)(X(tb_eigen_FMM) * F) {
    if (F->n < TB_EIGEN_BLOCKSIZE) {
        free(F->V);
        free(F->lambda);
    }
    else {
        X(destroy_hierarchicalmatrix)(F->F0);
        X(destroy_tb_eigen_FMM)(F->F1);
        X(destroy_tb_eigen_FMM)(F->F2);
        X(destroy_sparse)(F->S);
        free(F->X);
        free(F->Y);
        free(F->t1);
        free(F->t2);
        free(F->lambda);
        free(F->p1);
        free(F->p2);
    }
    free(F);
}

void X(destroy_tb_eigen_ADI)(X(tb_eigen_ADI) * F) {
    if (F->n < TB_EIGEN_BLOCKSIZE) {
        free(F->V);
        free(F->lambda);
    }
    else {
        X(destroy_lowrankmatrix)(F->F0);
        X(destroy_tb_eigen_ADI)(F->F1);
        X(destroy_tb_eigen_ADI)(F->F2);
        free(F->lambda);
    }
    free(F);
}

void X(destroy_symmetric_tridiagonal_qr)(X(symmetric_tridiagonal_qr) * F) {
    free(F->s);
    free(F->c);
    free(F->R);
    free(F);
}

void X(destroy_modified_plan)(X(modified_plan) * P) {
    if (P->nv < 1) {
        X(destroy_triangular_banded)(P->R);
    }
    else {
        X(destroy_triangular_banded)(P->K);
        X(destroy_triangular_banded)(P->R);
    }
    free(P);
}

size_t X(summary_size_tb_eigen_FMM)(X(tb_eigen_FMM) * F) {
    size_t S = 0;
    if (F->n < TB_EIGEN_BLOCKSIZE)
        S += sizeof(FLT)*F->n*(F->n+1);
    else {
        S += X(summary_size_hierarchicalmatrix)(F->F0);
        S += X(summary_size_tb_eigen_FMM)(F->F1);
        S += X(summary_size_tb_eigen_FMM)(F->F2);
        S += sizeof(FLT)*F->n*(2*F->b+1);
        S += sizeof(int)*F->n;
    }
    return S;
}

size_t X(summary_size_tb_eigen_ADI)(X(tb_eigen_ADI) * F) {
    size_t S = 0;
    if (F->n < TB_EIGEN_BLOCKSIZE)
        S += sizeof(FLT)*F->n*(F->n+1);
    else {
        S += X(summary_size_lowrankmatrix)(F->F0);
        S += X(summary_size_tb_eigen_ADI)(F->F1);
        S += X(summary_size_tb_eigen_ADI)(F->F2);
        S += sizeof(FLT)*F->n;
    }
    return S;
}

X(sparse) * X(malloc_sparse)(const int m, const int n, const int nnz) {
    X(sparse) * A = malloc(sizeof(X(sparse)));
    A->p = malloc(nnz*sizeof(int));
    A->q = malloc(nnz*sizeof(int));
    A->v = malloc(nnz*sizeof(FLT));
    A->m = m;
    A->n = n;
    A->nnz = nnz;
    return A;
}

X(sparse) * X(calloc_sparse)(const int m, const int n, const int nnz) {
    X(sparse) * A = malloc(sizeof(X(sparse)));
    A->p = calloc(nnz, sizeof(int));
    A->q = calloc(nnz, sizeof(int));
    A->v = calloc(nnz, sizeof(FLT));
    A->m = m;
    A->n = n;
    A->nnz = nnz;
    return A;
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

void X(realloc_triangular_banded)(X(triangular_banded) * A, const int b) {
    int n = A->n;
    FLT * data = calloc(n*(b+1), sizeof(FLT));
    if (b > A->b)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < A->b+1; i++)
                data[i+b-A->b+j*(b+1)] = A->data[i+j*(A->b+1)];
    else if (b < A->b)
        for (int j = 0; j < n; j++)
            for (int i = 0; i < b+1; i++)
                data[i+j*(b+1)] = A->data[i+A->b-b+j*(A->b+1)];
    free(A->data);
    A->data = data;
    A->b = b;
}

// Allocate a pointer without reallocating triangular banded data.
X(triangular_banded) * X(view_triangular_banded)(const X(triangular_banded) * A, const unitrange i) {
    X(triangular_banded) * V = malloc(sizeof(X(triangular_banded)));
    V->data = A->data + i.start*(A->b+1);
    V->n = i.stop-i.start;
    V->b = A->b;
    return V;
}

X(banded) * X(convert_triangular_banded_to_banded)(X(triangular_banded) * A) {
    X(banded) * B = malloc(sizeof(X(banded)));
    B->data = A->data;
    B->n = B->m = A->n;
    B->l = 0;
    B->u = A->b;
    free(A);
    return B;
}

X(triangular_banded) * X(convert_banded_to_triangular_banded)(X(banded) * A) {
    X(triangular_banded) * B = malloc(sizeof(X(triangular_banded)));
    if (A->l == 0) {
        B->data = A->data;
        B->n = A->n;
        B->b = A->u;
        free(A);
        return B;
    }
    else {
        B->data = calloc((A->u+1)*A->n, sizeof(FLT));
        for (int j = 0; j < A->n; j++)
            for (int i = 0; i < A->u+1+MIN(0, A->l); i++)
                B->data[i+j*(A->u+1)] = A->data[i+j*(A->l+A->u+1)];
        B->n = A->n;
        B->b = A->u;
        X(destroy_banded)(A);
        return B;
    }
}

X(symmetric_tridiagonal) * X(convert_banded_to_symmetric_tridiagonal)(X(banded) * A) {
    X(symmetric_tridiagonal) * T = malloc(sizeof(X(symmetric_tridiagonal)));
    int n = T->n = A->n;
    T->a = malloc(n*sizeof(FLT));
    T->b = malloc((n-1)*sizeof(FLT));
    for (int i = 0; i < n; i++)
        T->a[i] = X(get_banded_index)(A, i, i);
    for (int i = 0; i < n-1; i++)
        T->b[i] = X(get_banded_index)(A, i, i+1);
    X(destroy_banded)(A);
    return T;
}

X(triangular_banded) * X(create_I_triangular_banded)(const int n, const int b) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, b);
    for (int j = 0; j < n; j++)
        A->data[b+j*(b+1)] = 1;
    return A;
}

FLT X(get_banded_index)(const X(banded) * A, const int i, const int j) {
    FLT * data = A->data;
    int m = A->m, n = A->n, l = A->l, u = A->u;
    if (0 <= i && 0 <= j && -l <= j-i && j-i <= u && i < m && j < n)
        return data[u+i+j*(l+u)];
    else
        return 0;
}

void X(set_banded_index)(const X(banded) * A, const FLT v, const int i, const int j) {
    FLT * data = A->data;
    int m = A->m, n = A->n, l = A->l, u = A->u;
    if (0 <= i && 0 <= j && -l <= j-i && j-i <= u && i < m && j < n)
        data[u+i+j*(l+u)] = v;
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
    int m = A->m, n = A->n, l = A->l, u = A->u;
    for (int i = 0; i < m; i++)
        y[i] = beta*y[i];
    for (int i = 0; i < m; i++)
        for (int j = MAX(0, i-l); j < MIN(n, i+u+1); j++)
            y[i] += alpha*X(get_banded_index)(A, i, j)*x[j];
}

// C ← α*A*B + β*C
void X(gbmm)(FLT alpha, X(banded) * A, X(banded) * B, FLT beta, X(banded) * C) {
    FLT ab, c;
    int m = A->m, n = A->n, p = B->n;
    int l = C->l, u = C->u, l1 = A->l, u1 = A->u, l2 = B->l, u2 = B->u;
    if (C->m != m || B->m != n || C->n != p)
        exit_failure("gbmm: sizes are off.");
    if (C->l < l1+l2 || C->u < u1+u2)
        exit_failure("gbmm: bandwidths are off.");
    for (int j = 0; j < p; j++)
        for (int i = MAX(0, j-u); i < MIN(m, j+l+1); i++) {
            ab = 0;
            for (int k = MAX(MAX(0, i-l1), j-u2); k < MIN(MIN(n, i+u1+1), j+l2+1); k++)
                ab += X(get_banded_index)(A, i, k)*X(get_banded_index)(B, k, j);
            c = X(get_banded_index)(C, i, j);
            X(set_banded_index)(C, alpha*ab+beta*c, i, j);
        }
}

// B ← (α*A+β*I)*B, A tridiagonal, l and u are the effective lower and upper bandwidths of B -- entries of B outside these are 0.
void X(tridiagonal_banded_multiplication)(FLT alpha, X(banded) * A, FLT beta, X(banded) * B, const int l, const int u) {
    int m = B->m, n = B->n;
    if (A->m != A->n)
        exit_failure("tridiagonal_banded_multiplication: A not square.");
    if (m != n)
        exit_failure("tridiagonal_banded_multiplication: B not square.");
    if (A->m != m)
        exit_failure("tridiagonal_banded_multiplication: sizes are off.");
    if (A->l != 1 || A->u != 1)
        exit_failure("tridiagonal_banded_multiplication: A not tridiagonal.");
    if (B->l <= l || B->u <= u)
        exit_failure("tridiagonal_banded_multiplication: effective bandwidths too large.");
    for (int j = 0; j < n; j++) {
        int i = MAX(0, j-u-1);
        FLT t = (alpha*X(get_banded_index)(A, i, i) + beta)*X(get_banded_index)(B, i, j) + alpha*X(get_banded_index)(A, i, i+1)*X(get_banded_index)(B, i+1, j);
        for (; i < MIN(m, j+l+1); i++) {
            FLT s = t;
            t = alpha*X(get_banded_index)(A, i+1, i)*X(get_banded_index)(B, i, j) + (alpha*X(get_banded_index)(A, i+1, i+1) + beta)*X(get_banded_index)(B, i+1, j) + alpha*X(get_banded_index)(A, i+1, i+2)*X(get_banded_index)(B, i+2, j);
            X(set_banded_index)(B, s, i, j);
        }
        X(set_banded_index)(B, t, i, j);
    }
}

// C ← α*A+β*B
void X(banded_add)(FLT alpha, X(banded) * A, FLT beta, X(banded) * B, X(banded) * C) {
    int m = C->m, n = C->n, l = C->l, u = C->u;
    if (A->m != m || B->m != m || A->n != n || B->n != n)
        exit_failure("banded_add: sizes are off.");
    if (l < MAX(A->l, B->l) || C->u < MAX(A->u, B->u))
        exit_failure("banded_add: bandwidths are off.");
    for (int j = 0; j < n; j++)
        for (int i = MAX(0, j-u); i < MIN(m, j+l+1); i++)
            X(set_banded_index)(C, alpha*X(get_banded_index)(A, i, j) + beta*X(get_banded_index)(B, i, j), i, j);
}

// A ← α*A+β*I
void X(banded_uniform_scaling_add)(FLT alpha, X(banded) * A, FLT beta) {
    int m = A->m, n = A->n, l = A->l, u = A->u;
    if (m != n)
        exit_failure("banded_uniform_scaling_add: A is not square.");
    for (int j = 0; j < n; j++) {
        for (int i = MAX(0, j-u); i < j; i++)
            X(set_banded_index)(A, alpha*X(get_banded_index)(A, i, j), i, j);
        X(set_banded_index)(A, alpha*X(get_banded_index)(A, j, j) + beta, j, j);
        for (int i = j+1; i < MIN(m, j+l+1); i++)
            X(set_banded_index)(A, alpha*X(get_banded_index)(A, i, j), i, j);
    }
}

X(banded) * X(operator_orthogonal_polynomial_clenshaw)(const int n, const FLT * c, const int incc, const FLT * A, const FLT * B, const FLT * C, X(banded) * X, FLT phi0) {
    int m = X->m;
    X(banded) * Bk = X(calloc_banded)(m, m, n-1, n-1);
    X(banded) * Bk1 = X(calloc_banded)(m, m, n-1, n-1);
    X(banded) * Bk2 = X(calloc_banded)(m, m, n-1, n-1);
    X(banded) * Bt;
    for (int k = n-1; k >= 0; k--) {
        X(tridiagonal_banded_multiplication)(A[k], X, B[k], Bk, n-k-2, n-k-2);
        X(banded_uniform_scaling_add)(-C[k+1], Bk2, c[k*incc]);
        X(banded_add)(1, Bk, 1, Bk2, Bk);
        for (int i = 0; i < m*(2*n-1); i++)
            Bk2->data[i] = Bk->data[i];
        Bt = Bk2;
        Bk2 = Bk1;
        Bk1 = Bk;
        Bk = Bt;
    }
    X(banded_uniform_scaling_add)(phi0, Bk, 0);
    X(destroy_banded)(Bk1);
    X(destroy_banded)(Bk2);
    return Bk;
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

// x ← (A-γB)⁻¹*x, x ← (A-γB)⁻ᵀ*x
void X(tssv)(char TRANS, X(triangular_banded) * A, X(triangular_banded) * B, FLT gamma, FLT * x) {
    int n = A->n, b = MAX(A->b, B->b);
    FLT t;
    if (TRANS == 'N') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = i+1; k < MIN(i+b+1, n); k++)
                t += (X(get_triangular_banded_index)(A, i, k)-gamma*X(get_triangular_banded_index)(B, i, k))*x[k];
            x[i] = (x[i] - t)/(X(get_triangular_banded_index)(A, i, i)-gamma*X(get_triangular_banded_index)(B, i, i));
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = MAX(i-b, 0); k < i; k++)
                t += (X(get_triangular_banded_index)(A, k, i)-gamma*X(get_triangular_banded_index)(B, k, i))*x[k];
            x[i] = (x[i] - t)/(X(get_triangular_banded_index)(A, i, i)-gamma*X(get_triangular_banded_index)(B, i, i));
        }
    }
}

static inline void X(compute_elementary_transformation)(const int n, FLT * v) {
    for (int i = 1; i < n; i++)
        v[i] /= v[0];
}

static inline void X(apply_elementary_transformation)(const int n, const FLT * v, FLT * A) {
    for (int i = 1; i < n; i++)
        A[i] -= v[i]*A[0];
}

void X(banded_lufact)(X(banded) * A) {
    int n = A->n, l = A->l, u = A->u;
    if (A->m != n)
        exit_failure("banded_lufact: A is not square.");
    int nu = l+u+1;
    FLT * data = A->data;
    for (int j = 0; j < n; j++) {
        int lv = MIN(l+1, n-j+1);
        FLT * v = data+u+j*nu;
        X(compute_elementary_transformation)(lv, v);
        for (int k = 1; k <= MIN(u, n-j-1); k++)
            X(apply_elementary_transformation)(lv, v, data+u-k+(j+k)*nu);
    }
}

static inline void X(compute_symmetric_elementary_transformation)(const int n, FLT * v) {
    for (int i = 1; i < n; i++)
        v[i] /= v[0];
    if (v[0] < 0) warning("banded_cholfact: A is not positive-definite.");
    v[0] = Y(sqrt)(v[0]);
}

static inline void X(apply_symmetric_elementary_transformation)(const int n, const FLT * v, FLT * A) {
    for (int i = 1; i < n; i++)
        A[i] -= v[i]*A[0];
    A[0] /= v[0];
}

void X(banded_cholfact)(X(banded) * A) {
    int n = A->n, l = A->l, u = A->u;
    if (A->m != n)
        exit_failure("banded_cholfact: A is not square.");
    if (l != u)
        exit_failure("banded_cholfact: A is not symmetric.");
    int nu = l+u+1;
    FLT * data = A->data;
    for (int j = 0; j < n; j++) {
        int lv = MIN(l+1, n-j+1);
        FLT * v = data+u+j*nu;
        X(compute_symmetric_elementary_transformation)(lv, v);
        for (int k = 1; k <= MIN(u, n-j-1); k++)
            X(apply_symmetric_elementary_transformation)(lv, v, data+u-k+(j+k)*nu);
    }
}

static inline FLT X(compute_reflector)(const int n, FLT * v) {
    if (n < 1)
        return 0;
    FLT v0 = v[0];
    FLT nrmv = 0;
    for (int i = 0; i < n; i++)
        nrmv += v[i]*v[i];
    if (nrmv == 0)
        return 0;
    nrmv = Y(sqrt)(nrmv);
    FLT nu = Y(copysign)(nrmv, v0);
    v0 += nu;
    v[0] = -nu;
    for (int i = 1; i < n; i++)
        v[i] /= v0;
    return v0/nu;
}

static inline void X(apply_reflector)(const int n, const FLT * v, const FLT tau, FLT * A) {
    if (n < 1)
        return;
    FLT vA = A[0];
    for (int i = 1; i < n; i++)
        vA += v[i]*A[i];
    vA *= tau;
    A[0] -= vA;
    for (int i = 1; i < n; i++)
        A[i] -= v[i]*vA;
}

X(banded_qr) * X(banded_qrfact)(X(banded) * A) {
    int m = A->m, n = A->n, l = A->l, u = A->u;
    X(banded) * R = X(calloc_banded)(m, n, l, l+u);
    FLT * tau = calloc(MIN(m, n), sizeof(FLT));
    FLT * D = R->data;
    FLT * B = A->data;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < l+u+1; i++)
            D[l+i+j*(2*l+u+1)] = B[i+j*(l+u+1)];
    u = R->u;
    int nu = l+u+1;
    for (int j = 0; j < MIN(m, n); j++) {
        int lh = MIN(l+1, m-j);
        FLT * v = D+u+j*nu;
        tau[j] = X(compute_reflector)(lh, v);
        for (int k = 1; k <= MIN(u, n-j-1); k++)
            X(apply_reflector)(lh, v, tau[j], D+u-k+(j+k)*nu);
    }
    X(banded_qr) * F = malloc(sizeof(X(banded_qr)));
    F->factors = R;
    F->tau = tau;
    F->UPLO = 'U';
    return F;
}

static inline FLT X(compute_ql_reflector)(const int n, FLT * v) {
    if (n < 1)
        return 0;
    FLT vnm1 = v[n-1];
    FLT nrmv = 0;
    for (int i = 0; i < n; i++)
        nrmv += v[i]*v[i];
    if (nrmv == 0)
        return 0;
    nrmv = Y(sqrt)(nrmv);
    FLT nu = Y(copysign)(nrmv, vnm1);
    vnm1 += nu;
    v[n-1] = -nu;
    for (int i = 0; i < n-1; i++)
        v[i] /= vnm1;
    return vnm1/nu;
}

static inline void X(apply_ql_reflector)(const int n, const FLT * v, const FLT tau, FLT * A) {
    if (n < 1)
        return;
    FLT vA = A[n-1];
    for (int i = 0; i < n-1; i++)
        vA += v[i]*A[i];
    vA *= tau;
    A[n-1] -= vA;
    for (int i = 0; i < n-1; i++)
        A[i] -= v[i]*vA;
}

// Doesn't work for rectangular
X(banded_ql) * X(banded_qlfact)(X(banded) * A) {
    int m = A->m, n = A->n, l = A->l, u = A->u;
    X(banded) * L = X(calloc_banded)(m, n, l+u, u);
    FLT * tau = calloc(MIN(m, n), sizeof(FLT));
    FLT * D = L->data;
    FLT * B = A->data;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < l+u+1; i++)
            D[i+j*(l+2*u+1)] = B[i+j*(l+u+1)];
    l = L->l;
    int nu = l+u+1;
    for (int j = MIN(m, n) - 1; j >= 0; j--) {
        int lh = MIN(u+1, j+1);
        FLT * v = D+u+1-lh+j*nu;
        tau[j] = X(compute_ql_reflector)(lh, v);
        for (int k = 1; k <= MIN(l, j); k++)
            X(apply_ql_reflector)(lh, v, tau[j], D+u+1-lh+k+(j-k)*nu);
    }
    X(banded_ql) * F = malloc(sizeof(X(banded_ql)));
    F->factors = L;
    F->tau = tau;
    F->UPLO = 'L';
    return F;
}

// x ← Q*x, x ← Qᵀ*x
void X(bqmv)(char TRANS, struct X(banded_orthogonal_triangular) * F, FLT * x) {
    if (F->UPLO == 'U') {
        X(banded) * R = F->factors;
        FLT * D = R->data;
        FLT * tau = F->tau;
        int m = R->m, n = R->n, l = R->l, u = R->u;
        int nu = l+u+1;
        if (TRANS == 'N') {
            for (int j = MIN(m, n) - 1; j >= 0; j--) {
                int lh = MIN(l+1, m-j);
                X(apply_reflector)(lh, D+u+j*nu, tau[j], x+j);
            }
        }
        else if (TRANS == 'T') {
            for (int j = 0; j < MIN(m, n); j++) {
                int lh = MIN(l+1, m-j);
                X(apply_reflector)(lh, D+u+j*nu, tau[j], x+j);
            }
        }
    }
    else if (F->UPLO == 'L') {
        X(banded) * L = F->factors;
        FLT * D = L->data;
        FLT * tau = F->tau;
        int m = L->m, n = L->n, l = L->l, u = L->u;
        int nu = l+u+1;
        if (TRANS == 'N') {
            for (int j = 0; j < MIN(m, n); j++) {
                int lh = MIN(u+1, j+1);
                X(apply_ql_reflector)(lh, D+u+1-lh+j*nu, tau[j], x-lh+1+j);
            }
        }
        else if (TRANS == 'T') {
            for (int j = MIN(m, n) - 1; j >= 0; j--) {
                int lh = MIN(u+1, j+1);
                X(apply_ql_reflector)(lh, D+u+1-lh+j*nu, tau[j], x-lh+1+j);
            }
        }
    }
}

// A ← Qᵀ*A
void X(partial_bqmm)(struct X(banded_orthogonal_triangular) * F, int nu, int nv, X(banded) * A) {
    X(banded) * L = F->factors;
    FLT * D = L->data;
    FLT * tau = F->tau;
    int m = L->m, n = L->n, l = L->l, u = L->u;
    int kx = A->l+A->u+1;
    FLT * x = malloc((kx+u)*sizeof(FLT));
    for (int colA = 0; colA < A->n; colA++) {
        for (int k = 0; k < u; k++)
            x[k] = 0;
        for (int k = 0; k < kx; k++)
            x[k+u] = X(get_banded_index)(A, k+colA-A->u, colA);

        for (int j = MIN(MIN(m, n) - 1, colA+u+nu-1); j >= MAX(colA-A->u, 0); j--) {
            int lh = MIN(u+1, j+1);
            X(apply_ql_reflector)(lh, D+u+1-lh+j*(l+u+1), tau[j], x+u+nu+nv-1-lh+u+j-colA);
        }
        for (int k = 0; k < kx; k++)
            X(set_banded_index)(A, x[k+u], k+colA-A->u, colA);
    }
    free(x);
}


// x ← R*x, x ← Rᵀ*x
void X(brmv)(char TRANS, X(banded_qr) * F, FLT * x) {
    X(banded) * R = F->factors;
    int n = R->n, l = R->l, u = R->u;
    FLT * data = R->data, t;
    if (TRANS == 'N') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = i; k < MIN(i+u+1, n); k++)
                t += data[u+i+k*(l+u)]*x[k];
            x[i] = t;
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = MAX(i-u, 0); k <= i; k++)
                t += data[u+k+i*(l+u)]*x[k];
            x[i] = t;
        }
    }
}

// x ← R⁻¹*x, x ← R⁻ᵀ*x
void X(brsv)(char TRANS, X(banded_qr) * F, FLT * x) {
    X(banded) * R = F->factors;
    int n = R->n, l = R->l, u = R->u;
    FLT * data = R->data, t;
    if (TRANS == 'N') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = i+1; k < MIN(i+u+1, n); k++)
                t += data[u+i+k*(l+u)]*x[k];
            x[i] = (x[i] - t)/data[u+i+i*(l+u)];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = MAX(i-u, 0); k < i; k++)
                t += data[u+k+i*(l+u)]*x[k];
            x[i] = (x[i] - t)/data[u+i+i*(l+u)];
        }
    }
}

// x ← L*x, x ← Lᵀ*x
void X(blmv)(char TRANS, X(banded_ql) * F, FLT * x) {
    X(banded) * L = F->factors;
    int n = L->n, l = L->l, u = L->u;
    FLT * data = L->data, t;
    if (TRANS == 'N') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = MAX(i-l, 0); k <= i; k++)
                t += data[u+i+k*(l+u)]*x[k];
            x[i] = t;
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = i; k < MIN(i+l+1, n); k++)
                t += data[u+k+i*(l+u)]*x[k];
            x[i] = t;
        }
    }
}

// x ← L⁻¹*x, x ← L⁻ᵀ*x
void X(blsv)(char TRANS, X(banded_ql) * F, FLT * x) {
    X(banded) * L = F->factors;
    int n = L->n, l = L->l, u = L->u;
    FLT * data = L->data, t;
    if (TRANS == 'N') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = MAX(i-l, 0); k < i; k++)
                t += data[u+i+k*(l+u)]*x[k];
            x[i] = (x[i] - t)/data[u+i+i*(l+u)];
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            t = 0;
            for (int k = i+1; k < MIN(i+l+1, n); k++)
                t += data[u+k+i*(l+u)]*x[k];
            x[i] = (x[i] - t)/data[u+i+i*(l+u)];
        }
    }
}

// AV = BVΛ

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda) {
    for (int j = 0; j < A->n; j++)
        lambda[j] = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
}

// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V) {
    int n = A->n, b = MAX(A->b, B->b);
    FLT t, kt, d, kd, lam;
    for (int j = 1; j < n; j++) {
        lam = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
        for (int i = j-1; i >= 0; i--) {
            t = kt = 0;
            for (int k = i+1; k < MIN(i+b+1, n); k++) {
                t += (X(get_triangular_banded_index)(A, i, k) - lam*X(get_triangular_banded_index)(B, i, k))*V[k+j*n];
                kt += (Y(fabs)(X(get_triangular_banded_index)(A, i, k)) + Y(fabs)(lam*X(get_triangular_banded_index)(B, i, k)))*Y(fabs)(V[k+j*n]);
            }
            d = lam*X(get_triangular_banded_index)(B, i, i) - X(get_triangular_banded_index)(A, i, i);
            kd = Y(fabs)(lam*X(get_triangular_banded_index)(B, i, i)) + Y(fabs)(X(get_triangular_banded_index)(A, i, i));
            if (Y(fabs)(d) < 4*kd*Y(eps)() || Y(fabs)(t) < 4*kt*Y(eps)())
                V[i+j*n] = 0;
            else
                V[i+j*n] = t/d;
        }
    }
}

// AV + BVΛ = CVΛ²

void X(triangular_banded_quadratic_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, X(triangular_banded) * C, FLT * lambda) {
    FLT a, b, c;
    for (int j = 0; j < A->n; j++) {
        a = X(get_triangular_banded_index)(A, j, j);
        b = X(get_triangular_banded_index)(B, j, j);
        c = X(get_triangular_banded_index)(C, j, j);
        lambda[j] = (b+Y(sqrt)(b*b+4*a*c))/(2*c);
    }
}

// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void X(triangular_banded_quadratic_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, X(triangular_banded) * C, FLT * V) {
    int n = A->n, bnd = MAX(MAX(A->b, B->b), C->b);
    FLT a, b, c, d, kd, t, kt, lam;
    for (int j = 1; j < n; j++) {
        a = X(get_triangular_banded_index)(A, j, j);
        b = X(get_triangular_banded_index)(B, j, j);
        c = X(get_triangular_banded_index)(C, j, j);
        lam = (b+Y(sqrt)(b*b+4*a*c))/(2*c);
        for (int i = j-1; i >= 0; i--) {
            t = kt = 0;
            for (int k = i+1; k < MIN(i+bnd+1, n); k++) {
                t += (X(get_triangular_banded_index)(A, i, k) + lam*(X(get_triangular_banded_index)(B, i, k) - lam*X(get_triangular_banded_index)(C, i, k)))*V[k+j*n];
                kt += (Y(fabs)(X(get_triangular_banded_index)(A, i, k)) + Y(fabs)(lam*(Y(fabs)(X(get_triangular_banded_index)(B, i, k)) + Y(fabs)(lam*X(get_triangular_banded_index)(C, i, k)))))*Y(fabs)(V[k+j*n]);
            }
            d = lam*(lam*X(get_triangular_banded_index)(C, i, i) - X(get_triangular_banded_index)(B, i, i)) - X(get_triangular_banded_index)(A, i, i);
            kd = Y(fabs)(lam*(Y(fabs)(lam*X(get_triangular_banded_index)(C, i, i)) + Y(fabs)(X(get_triangular_banded_index)(B, i, i)))) + Y(fabs)(X(get_triangular_banded_index)(A, i, i));
            if (Y(fabs)(d) < 4*kd*Y(eps)() && Y(fabs)(t) < 4*kt*Y(eps)())
                V[i+j*n] = 0;
            else
                V[i+j*n] = t/d;
        }
    }
}

// Assumptions: x, y are non-decreasing.
static inline int X(count_intersections)(const int m, const FLT * x, const int n, const FLT * y, const FLT epsilon) {
    int istart = 0, idx = 0;
    for (int j = 0; j < n; j++) {
        int i = istart;
        int thefirst = 1;
        while (i < m) {
            if (Y(fabs)(x[i] - y[j]) < epsilon*MAX(Y(fabs)(x[i]), Y(fabs)(y[j]))) {
                idx++;
                if (thefirst) {
                    istart = i;
                    thefirst--;
                }
            }
            else if (x[i] > y[j])
                break;
            i++;
        }
    }
    return idx;
}

// Assumptions: p and q have been malloc'ed with `idx` integers.
static inline void X(produce_intersection_indices)(const int m, const FLT * x, const int n, const FLT * y, const FLT epsilon, int * p, int * q) {
    int istart = 0, idx = 0;
    for (int j = 0; j < n; j++) {
        int i = istart;
        int thefirst = 1;
        while (i < m) {
            if (Y(fabs)(x[i] - y[j]) < epsilon*MAX(Y(fabs)(x[i]), Y(fabs)(y[j]))) {
                p[idx] = i;
                q[idx] = j;
                idx++;
                if (thefirst) {
                    istart = i;
                    thefirst--;
                }
            }
            else if (x[i] > y[j])
                break;
            i++;
        }
    }
}

static inline X(sparse) * X(get_sparse_from_eigenvectors)(X(tb_eigen_FMM) * F1, X(triangular_banded) * A, X(triangular_banded) * B, FLT * D, int * p1, int * p2, int * p3, int * p4, int n, int s, int b, int idx) {
    X(sparse) * S = X(malloc_sparse)(s, n-s, idx);
    FLT * V = calloc(n, sizeof(FLT));
    for (int l = 0; l < idx; l++) {
        int j = p2[p4[l]]+s;
        for (int i = 0; i < n; i++)
            V[i] = 0;
        V[j] = D[j];
        FLT t, kt, d, kd, lam;
        lam = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
        for (int i = j-1; i >= 0; i--) {
            t = kt = 0;
            for (int k = i+1; k < MIN(i+b+1, n); k++) {
                t += (X(get_triangular_banded_index)(A, i, k) - lam*X(get_triangular_banded_index)(B, i, k))*V[k];
                kt += (Y(fabs)(X(get_triangular_banded_index)(A, i, k)) + Y(fabs)(lam*X(get_triangular_banded_index)(B, i, k)))*Y(fabs)(V[k]);
            }
            d = lam*X(get_triangular_banded_index)(B, i, i) - X(get_triangular_banded_index)(A, i, i);
            kd = Y(fabs)(lam*X(get_triangular_banded_index)(B, i, i)) + Y(fabs)(X(get_triangular_banded_index)(A, i, i));
            if (Y(fabs)(d) < 4*kd*Y(eps)() || Y(fabs)(t) < 4*kt*Y(eps)())
                V[i] = 0;
            else
                V[i] = t/d;
        }
        X(bfsv)('N', F1, V);
        S->p[l] = p1[p3[l]];
        S->q[l] = p2[p4[l]];
        S->v[l] = V[p1[p3[l]]];
    }
    free(V);
    return S;
}

X(tb_eigen_FMM) * X(tb_eig_FMM)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * D) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    X(tb_eigen_FMM) * F = malloc(sizeof(X(tb_eigen_FMM)));
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT * V = calloc(n*n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            V[i+i*n] = D[i];
        F->lambda = malloc(n*sizeof(FLT));
        X(triangular_banded_eigenvalues)(A, B, F->lambda);
        X(triangular_banded_eigenvectors)(A, B, V);
        F->V = V;
        F->n = n;
        F->b = b;
    }
    else {
        int s = n>>1;
        unitrange i = {0, s}, j = {s, n};
        X(triangular_banded) * A1 = X(view_triangular_banded)(A, i);
        X(triangular_banded) * B1 = X(view_triangular_banded)(B, i);
        X(triangular_banded) * A2 = X(view_triangular_banded)(A, j);
        X(triangular_banded) * B2 = X(view_triangular_banded)(B, j);

        F->F1 = X(tb_eig_FMM)(A1, B1, D);
        F->F2 = X(tb_eig_FMM)(A2, B2, D+s);

        FLT * lambda = malloc(n*sizeof(FLT));
        for (int i = 0; i < s; i++)
            lambda[i] = F->F1->lambda[i];
        for (int i = 0; i < n-s; i++)
            lambda[i+s] = F->F2->lambda[i];

        FLT * X = calloc(s*b, sizeof(FLT));
        for (int j = 0; j < b; j++) {
            X[s-b+j+j*s] = 1;
            X(tbsv)('N', B1, X+j*s);
            X(bfsv)('N', F->F1, X+j*s);
        }

        FLT * Y = calloc((n-s)*b, sizeof(FLT));
        for (int j = 0; j < b1; j++)
            for (int k = j; k < b1; k++)
                Y[j+k*(n-s)] = X(get_triangular_banded_index)(A, k+s-b1, j+s);
        FLT * Y2 = calloc((n-s)*b2, sizeof(FLT));
        for (int j = 0; j < b2; j++)
            for (int k = j; k < b2; k++)
                Y2[j+k*(n-s)] = X(get_triangular_banded_index)(B, k+s-b2, j+s);

        for (int j = 0; j < b1; j++)
            X(bfmv)('T', F->F2, Y+j*(n-s));
        for (int j = 0; j < b2; j++)
            X(bfmv)('T', F->F2, Y2+j*(n-s));

        for (int j = 0; j < b2; j++)
            for (int i = 0; i < n-s; i++)
                Y[i+(j+b-b2)*(n-s)] = Y[i+(j+b-b2)*(n-s)]-lambda[i+s]*Y2[i+j*(n-s)];

        int * p1 = malloc(s*sizeof(int));
        for (int i = 0; i < s; i++)
            p1[i] = i;
        X(quicksort_1arg)(lambda, p1, 0, s-1, X(lt));
        int * p2 = malloc((n-s)*sizeof(int));
        for (int i = 0; i < n-s; i++)
            p2[i] = i;
        X(quicksort_1arg)(lambda+s, p2, 0, n-s-1, X(lt));

        int idx = X(count_intersections)(s, lambda, n-s, lambda+s, 16*Y(sqrt)(Y(eps)()));
        int * p3 = malloc(idx*sizeof(int));
        int * p4 = malloc(idx*sizeof(int));
        X(produce_intersection_indices)(s, lambda, n-s, lambda+s, 16*Y(sqrt)(Y(eps)()), p3, p4);
        X(sparse) * S = X(get_sparse_from_eigenvectors)(F->F1, A, B, D, p1, p2, p3, p4, n, s, b, idx);
        free(p3);
        free(p4);

        F->F0 = X(sample_hierarchicalmatrix)(X(thresholded_cauchykernel), lambda, lambda, i, j, 'G');
        F->X = X;
        F->Y = Y;
        F->S = S;
        F->t1 = calloc(s*FT_GET_MAX_THREADS(), sizeof(FLT));
        F->t2 = calloc((n-s)*FT_GET_MAX_THREADS(), sizeof(FLT));
        X(perm)('T', lambda, p1, s);
        X(perm)('T', lambda+s, p2, n-s);
        F->lambda = lambda;
        F->p1 = p1;
        F->p2 = p2;
        F->n = n;
        F->b = b;
        free(A1);
        free(B1);
        free(A2);
        free(B2);
        free(Y2);
    }
    return F;
}

X(tb_eigen_ADI) * X(tb_eig_ADI)(X(triangular_banded) * A, X(triangular_banded) * B) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    X(tb_eigen_ADI) * F = malloc(sizeof(X(tb_eigen_ADI)));
    if (n < TB_EIGEN_BLOCKSIZE) {
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
        int s = n>>1;
        unitrange i = {0, s}, j = {s, n};
        X(triangular_banded) * A1 = X(view_triangular_banded)(A, i);
        X(triangular_banded) * B1 = X(view_triangular_banded)(B, i);
        X(triangular_banded) * A2 = X(view_triangular_banded)(A, j);
        X(triangular_banded) * B2 = X(view_triangular_banded)(B, j);

        F->F1 = X(tb_eig_ADI)(A1, B1);
        F->F2 = X(tb_eig_ADI)(A2, B2);

        FLT * lambda = malloc(n*sizeof(FLT));
        for (int i = 0; i < s; i++)
            lambda[i] = F->F1->lambda[i];
        for (int i = 0; i < n-s; i++)
            lambda[i+s] = F->F2->lambda[i];

        FLT * X = calloc(s*b, sizeof(FLT));
        #pragma omp parallel for num_threads(MIN(b, FT_GET_MAX_THREADS()))
        for (int j = 0; j < b; j++) {
            X[s-b+j+j*s] = -1;
            X(tbsv)('N', B1, X+j*s);
            X(bfsv_ADI)('N', F->F1, X+j*s);
        }

        FLT * Y = calloc((n-s)*b, sizeof(FLT));
        for (int j = 0; j < b1; j++)
            for (int k = j; k < b1; k++)
                Y[j+k*(n-s)] = X(get_triangular_banded_index)(A, k+s-b1, j+s);
        FLT * Y2 = calloc((n-s)*b2, sizeof(FLT));
        for (int j = 0; j < b2; j++)
            for (int k = j; k < b2; k++)
                Y2[j+k*(n-s)] = X(get_triangular_banded_index)(B, k+s-b2, j+s);

        #pragma omp parallel for num_threads(MIN(b1+b2, FT_GET_MAX_THREADS()))
        for (int j = 0; j < b1+b2; j++)
            if (j < b1)
                X(bfmv_ADI)('T', F->F2, Y+j*(n-s));
            else
                X(bfmv_ADI)('T', F->F2, Y2+(j-b1)*(n-s));

        for (int j = 0; j < b2; j++)
            for (int i = 0; i < n-s; i++)
                Y[i+(j+b-b2)*(n-s)] = Y[i+(j+b-b2)*(n-s)]-lambda[i+s]*Y2[i+j*(n-s)];

        F->F0 = X(ddfadi)(s, lambda, n-s, lambda+s, b, X, Y);
        F->lambda = lambda;
        F->n = n;
        F->b = b;
        free(A1);
        free(B1);
        free(A2);
        free(B2);
        free(Y2);
    }
    return F;
}

void X(scale_rows_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT * V = F->V;
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= alpha*x[i];
    }
    else {
        int s = n>>1;
        X(scale_rows_tb_eigen_FMM)(alpha, x, F->F1);
        X(scale_rows_tb_eigen_FMM)(alpha, x+s, F->F2);
    }
}

void X(scale_rows_tb_eigen_ADI)(FLT alpha, FLT * x, X(tb_eigen_ADI) * F) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT * V = F->V;
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= alpha*x[i];
    }
    else {
        int s = n>>1;
        X(scale_rows_tb_eigen_ADI)(alpha, x, F->F1);
        X(scale_rows_tb_eigen_ADI)(alpha, x+s, F->F2);
    }
}

void X(scale_columns_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT scl, * V = F->V;
        for (int j = 0; j < n; j++) {
            scl = alpha*x[j];
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= scl;
        }
    }
    else {
        int s = n>>1, b = F->b;
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

void X(scale_columns_tb_eigen_ADI)(FLT alpha, FLT * x, X(tb_eigen_ADI) * F) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT scl, * V = F->V;
        for (int j = 0; j < n; j++) {
            scl = alpha*x[j];
            for (int i = 0; i <= j; i++)
                V[i+j*n] *= scl;
        }
    }
    else {
        int s = n>>1, b = F->b;
        X(scale_columns_lowrankmatrix)(1, x+s, F->F0);
        FLT * t = malloc(s*sizeof(FLT));
        for (int i = 0; i < s; i++)
            t[i] = 1/x[i];
        X(scale_rows_lowrankmatrix)(1, t, F->F0);
        free(t);
        X(scale_columns_tb_eigen_ADI)(alpha, x, F->F1);
        X(scale_columns_tb_eigen_ADI)(alpha, x+s, F->F2);
    }
}

// x ← A*x, x ← Aᵀ*x
void X(trmv)(char TRANS, int n, FLT * A, int LDA, FLT * x) {
    if (TRANS == 'N') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++)
                x[i] += A[i+j*LDA]*x[j];
            x[j] *= A[j+j*LDA];
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            x[i] *= A[i+i*LDA];
            for (int j = i-1; j >= 0; j--)
                x[i] += A[j+i*LDA]*x[j];
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(trsv)(char TRANS, int n, FLT * A, int LDA, FLT * x) {
    if (TRANS == 'N') {
        for (int j = n-1; j >= 0; j--) {
            x[j] /= A[j+j*LDA];
            for (int i = 0; i < j; i++)
                x[i] -= A[i+j*LDA]*x[j];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++)
                x[i] -= A[j+i*LDA]*x[j];
            x[i] /= A[i+i*LDA];
        }
    }
}

// B ← A*B, B ← Aᵀ*B
#if defined(FT_USE_CBLAS_S)
    void X(trmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
        if (TRANS == 'N')
            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
        else if (TRANS == 'T')
            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
    }
#elif defined(FT_USE_CBLAS_D)
    void X(trmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
        if (TRANS == 'N')
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
        else if (TRANS == 'T')
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
    }
#else
    void X(trmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
        #pragma omp parallel for
        for (int j = 0; j < N; j++)
            X(trmv)(TRANS, n, A, LDA, B+j*LDB);
    }
#endif

// B ← A⁻¹*B, B ← A⁻ᵀ*B
#if defined(FT_USE_CBLAS_S)
    void X(trsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
        if (TRANS == 'N')
            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
        else if (TRANS == 'T')
            cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
    }
#elif defined(FT_USE_CBLAS_D)
    void X(trsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
        if (TRANS == 'N')
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
        else if (TRANS == 'T')
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, n, N, 1, A, LDA, B, LDB);
    }
#else
    void X(trsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
        #pragma omp parallel for
        for (int j = 0; j < N; j++)
            X(trsv)(TRANS, n, A, LDA, B+j*LDB);
    }
#endif

// x ← A*x, x ← Aᵀ*x
void X(bfmv)(char TRANS, X(tb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        X(trmv)(TRANS, n, F->V, n, x);
    else {
        int s = n>>1, b = F->b;
        FLT * t1 = F->t1+s*FT_GET_THREAD_NUM(), * t2 = F->t2+(n-s)*FT_GET_THREAD_NUM();
        int * p1 = F->p1, * p2 = F->p2;
        X(sparse) * S = F->S;
        if (TRANS == 'N') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    t2[i] = F->Y[p2[i]+k*(n-s)]*x[p2[i]+s];
                X(ghmv)(TRANS, -1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[p1[i]] += t1[i]*F->X[p1[i]+k*s];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->p[l]] += S->v[l]*x[S->q[l]+s];
            X(bfmv)(TRANS, F->F1, x);
            X(bfmv)(TRANS, F->F2, x+s);
        }
        else if (TRANS == 'T') {
            X(bfmv)(TRANS, F->F1, x);
            X(bfmv)(TRANS, F->F2, x+s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    t1[i] = F->X[p1[i]+k*s]*x[p1[i]];
                X(ghmv)(TRANS, -1, F->F0, t1, 0, t2);
                for (int i = 0; i < n-s; i++)
                    x[p2[i]+s] += t2[i]*F->Y[p2[i]+k*(n-s)];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->q[l]+s] += S->v[l]*x[S->p[l]];
        }
    }
}

// x ← A*x, x ← Aᵀ*x
void X(bfmv_ADI)(char TRANS, X(tb_eigen_ADI) * F, FLT * x) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        X(trmv)(TRANS, n, F->V, n, x);
    else {
        int s = n>>1, b = F->b;
        if (TRANS == 'N') {
            X(lrmv)(TRANS, 1, F->F0, x+s, 1, x);
            X(bfmv_ADI)(TRANS, F->F1, x);
            X(bfmv_ADI)(TRANS, F->F2, x+s);
        }
        else if (TRANS == 'T') {
            X(bfmv_ADI)(TRANS, F->F1, x);
            X(bfmv_ADI)(TRANS, F->F2, x+s);
            X(lrmv)(TRANS, 1, F->F0, x, 1, x+s);
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(bfsv)(char TRANS, X(tb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        X(trsv)(TRANS, n, F->V, n, x);
    else {
        int s = n>>1, b = F->b;
        FLT * t1 = F->t1+s*FT_GET_THREAD_NUM(), * t2 = F->t2+(n-s)*FT_GET_THREAD_NUM();
        int * p1 = F->p1, * p2 = F->p2;
        X(sparse) * S = F->S;
        if (TRANS == 'N') {
            X(bfsv)(TRANS, F->F1, x);
            X(bfsv)(TRANS, F->F2, x+s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    t2[i] = F->Y[p2[i]+k*(n-s)]*x[p2[i]+s];
                X(ghmv)(TRANS, 1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[p1[i]] += t1[i]*F->X[p1[i]+k*s];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->p[l]] -= S->v[l]*x[S->q[l]+s];
        }
        else if (TRANS == 'T') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    t1[i] = F->X[p1[i]+k*s]*x[p1[i]];
                X(ghmv)(TRANS, 1, F->F0, t1, 0, t2);
                for (int i = 0; i < n-s; i++)
                    x[p2[i]+s] += t2[i]*F->Y[p2[i]+k*(n-s)];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->q[l]+s] -= S->v[l]*x[S->p[l]];
            X(bfsv)(TRANS, F->F1, x);
            X(bfsv)(TRANS, F->F2, x+s);
        }
    }
}

// x ← A*x, x ← Aᵀ*x
void X(bfsv_ADI)(char TRANS, X(tb_eigen_ADI) * F, FLT * x) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        X(trsv)(TRANS, n, F->V, n, x);
    else {
        int s = n>>1, b = F->b;
        if (TRANS == 'N') {
            X(bfsv_ADI)(TRANS, F->F1, x);
            X(bfsv_ADI)(TRANS, F->F2, x+s);
            X(lrmv)(TRANS, -1, F->F0, x+s, 1, x);
        }
        else if (TRANS == 'T') {
            X(lrmv)(TRANS, -1, F->F0, x, 1, x+s);
            X(bfsv_ADI)(TRANS, F->F1, x);
            X(bfsv_ADI)(TRANS, F->F2, x+s);
        }
    }
}

void X(bfmm)(char TRANS, X(tb_eigen_FMM) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bfmv)(TRANS, F, B+j*LDB);
}

void X(bfsm)(char TRANS, X(tb_eigen_FMM) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bfsv)(TRANS, F, B+j*LDB);
}

void X(bfmm_ADI)(char TRANS, X(tb_eigen_ADI) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bfmv_ADI)(TRANS, F, B+j*LDB);
}

void X(bfsm_ADI)(char TRANS, X(tb_eigen_ADI) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bfsv_ADI)(TRANS, F, B+j*LDB);
}

// ‖V‖₂ ≤ √‖V‖₁‖V‖∞
static inline FLT X(normest_dense)(FLT * V, int n) {
    FLT R = 0, C = 0, t;
    for (int j = 0; j < n; j++) {
        t = 0;
        for (int i = 0; i < n; i++)
            t += Y(fabs)(V[i+j*n]);
        if (t > C)
            C = t;
        t = 0;
        for (int i = 0; i < n; i++)
            t += Y(fabs)(V[j+i*n]);
        if (t > R)
            R = t;
    }
    return Y(sqrt)(R*C);
}

// ‖V‖₂ = ‖(V₁₁ 0)(I V₁₂)‖
//        ‖(0 V₂₂)(0  I )‖ ≤ max{‖V₁₁‖₂, ‖V₂₂‖₂} (1 + ‖V₁₂‖_F)
FLT X(normest_tb_eigen_ADI)(X(tb_eigen_ADI) * F) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        return X(normest_dense)(F->V, n);
    else
        return MAX(X(normest_tb_eigen_ADI)(F->F1), X(normest_tb_eigen_ADI)(F->F2))*(1 + X(norm_lowrankmatrix)(F->F0));
}

static inline void X(compute_givens)(const FLT x, const FLT y, FLT * c, FLT * s, FLT * r) {
    * r = Y(hypot)(x, y);
    if (* r <= Y(floatmin)()/Y(eps)()) {
        * c = 1;
        * s = 0;
    }
    else {
        * c = x / * r;
        * s = y / * r;
    }
}

// Computes a Givens rotation QR factorization of A.
// Q = G[0]G[1]⋯G[n-2]
X(symmetric_tridiagonal_qr) * X(symmetric_tridiagonal_qrfact)(X(symmetric_tridiagonal) * A) {
    int n = A->n;
    FLT * a = A->a;
    FLT * b = A->b;
    FLT r;
    FLT aip1 = a[0];
    FLT bip1r = b[0];
    FLT * s = malloc((n-1)*sizeof(FLT));
    FLT * c = malloc((n-1)*sizeof(FLT));
    X(banded) * R = X(calloc_banded)(n, n, 0, 2);
    for (int i = 0; i < n-2; i++) {
        X(compute_givens)(aip1, -b[i], c+i, s+i, &r);
        X(set_banded_index)(R, r, i, i);
        X(set_banded_index)(R, c[i]*bip1r-s[i]*a[i+1], i, i+1);
        X(set_banded_index)(R, -s[i]*b[i+1], i, i+2);
        aip1 = c[i]*a[i+1]+s[i]*bip1r;
        bip1r = c[i]*b[i+1];
    }
    X(compute_givens)(aip1, -b[n-2], c+n-2, s+n-2, &r);
    X(set_banded_index)(R, r, n-2, n-2);
    X(set_banded_index)(R, c[n-2]*bip1r-s[n-2]*a[n-1], n-2, n-1);
    X(set_banded_index)(R, c[n-2]*a[n-1]+s[n-2]*bip1r, n-1, n-1);

    X(symmetric_tridiagonal_qr) * F = malloc(sizeof(X(symmetric_tridiagonal_qr)));
    F->s = s;
    F->c = c;
    F->n = n;
    F->R = R;
    return F;
}

void X(mpmv)(char TRANS, X(modified_plan) * P, FLT * x) {
    if (P->nv < 1) {
        X(tbsv)(TRANS, P->R, x);
    }
    else {
        if (TRANS == 'N') {
            X(tbsv)('N', P->K, x);
            X(tbmv)('N', P->R, x);
        }
        else if (TRANS == 'T') {
            X(tbmv)('T', P->R, x);
            X(tbsv)('T', P->K, x);
        }
    }
}

void X(mpsv)(char TRANS, X(modified_plan) * P, FLT * x) {
    if (P->nv < 1) {
        X(tbmv)(TRANS, P->R, x);
    }
    else {
        if (TRANS == 'N') {
            X(tbsv)('N', P->R, x);
            X(tbmv)('N', P->K, x);
        }
        else if (TRANS == 'T') {
            X(tbmv)('T', P->K, x);
            X(tbsv)('T', P->R, x);
        }
    }
}

void X(mpmm)(char TRANS, X(modified_plan) * P, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(mpmv)(TRANS, P, B+j*LDB);
}

void X(mpsm)(char TRANS, X(modified_plan) * P, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(mpsv)(TRANS, P, B+j*LDB);
}

X(modified_plan) * X(plan_modified)(const int n, X(banded) * (*operator_clenshaw)(const int n, const int nc, const FLT * c, const int incc, const X(cop_params) params), const X(cop_params) params, const int nu, const FLT * u, const int nv, const FLT * v, const int verbose) {
    if (nv < 1) {
        // polynomial case
        X(banded) * U = operator_clenshaw(n, nu, u, 1, params);
        X(banded_cholfact)(U);
        X(triangular_banded) * R = X(convert_banded_to_triangular_banded)(U);
        X(modified_plan) * P = malloc(sizeof(X(modified_plan)));
        P->R = R;
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        return P;
    }
    else {
        // rational case
        X(banded_ql) * F;
        int N = 2*n;
        while (1) {
            X(banded) * V = operator_clenshaw(N+nu+nv, nv, v, 1, params);

            FLT nrm_Vb = 0;
            FLT * Vb = calloc(N*(nv-1), sizeof(FLT));
            for (int j = 0; j < nv-1; j++)
                for (int i = N-nv+1+j; i < N; i++) {
                    Vb[i+j*N] = X(get_banded_index)(V, i, j+N);
                    nrm_Vb += Vb[i+j*N]*Vb[i+j*N];
                }
            nrm_Vb = Y(sqrt)(nrm_Vb);

            // truncate it for QL
            V->m = V->n = N;
            F = X(banded_qlfact)(V);

            for (int j = 0; j < nv-1; j++)
                X(bqmv)('T', F, Vb+j*N);

            FLT nrm_Vn = 0;
            for (int j = 0; j < nv-1; j++)
                for (int i = 0; i < n; i++)
                    nrm_Vn += Vb[i+j*N]*Vb[i+j*N];
            nrm_Vn = Y(sqrt)(nrm_Vn);

            free(Vb);
            X(destroy_banded)(V);
            if (N > FT_MODIFIED_NMAX) {
                warning("plan_modified: dimension of QL factorization, N, exceeds maximum allowable.");
                break;
            }
            else if (nv*nrm_Vn <= Y(eps)()*nrm_Vb) {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %4.3e ≤ %4.3e\n", N, (double) nv*nrm_Vn, (double) Y(eps)()*nrm_Vb);
                break;
            }
            else {
                verbose && printf("N = %i, and the bound on the relative 2-norm: %4.3e ≰ %4.3e\n", N, (double) nv*nrm_Vn, (double) Y(eps)()*nrm_Vb);
            }
            X(destroy_banded_ql)(F);
            N <<= 1;
        }
        F->factors->m = F->factors->n = n+nu+nv;
        X(banded) * U = operator_clenshaw(n+nu+nv, nu, u, 1, params);

        X(banded) * Lt = X(calloc_banded)(n+nu+nv, n+nu+nv, 0, F->factors->l);
        for (int j = 0; j < n+nu+nv; j++)
            for (int i = j; i < MIN(n+nu+nv, j+F->factors->l+1); i++)
                X(set_banded_index)(Lt, X(get_banded_index)(F->factors, i, j), j, i);

        FLT * D = calloc(n+nu+nv, sizeof(FLT));
        for (int j = 0; j < n+nu+nv; j++) {
            D[j] = (Y(signbit)(X(get_banded_index)(Lt, j, j))) ? -1 : 1;
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                X(set_banded_index)(Lt, X(get_banded_index)(Lt, i, j)*D[j], i, j);
        }

        X(banded) * ULt = X(calloc_banded)(n+nu+nv, n+nu+nv, nu+nv-2, nu+2*nv-3);
        X(gbmm)(1, U, Lt, 0, ULt);
        // ULᵀ ← QᵀULᵀ
        X(partial_bqmm)(F, nu, nv, ULt);

        int b = nu+nv-2;
        X(banded) * QtULt = X(calloc_banded)(n, n, b, b);
        for (int i = 0; i < n; i++)
            for (int j = MAX(i-b, 0); j < MIN(i+b+1, n); j++)
                X(set_banded_index)(QtULt, D[i]*X(get_banded_index)(ULt, i, j), i, j);
        X(banded_cholfact)(QtULt);
        X(triangular_banded) * K = X(convert_banded_to_triangular_banded)(QtULt);

        X(triangular_banded) * R = X(calloc_triangular_banded)(n, Lt->u);
        for (int j = 0; j < n; j++)
            for (int i = j; i >= MAX(0, j-Lt->u); i--)
                X(set_triangular_banded_index)(R, X(get_banded_index)(Lt, i, j), i, j);

        free(D);
        X(destroy_banded)(U);
        X(destroy_banded)(Lt);
        X(destroy_banded)(ULt);
        X(destroy_banded_ql)(F);
        X(modified_plan) * P = malloc(sizeof(X(modified_plan)));
        P->n = n;
        P->nu = nu;
        P->nv = nv;
        P->K = K;
        P->R = R;
        return P;
    }
}

void Y(execute_jacobi_similarity)(const X(modified_plan) * P, const int n, const FLT * ap, const FLT * bp, FLT * aq, FLT * bq) {
    if (P->nv < 1) {
        // P = Q R => XQ = R XP R⁻¹, but we can calculate it only up to n-1. R_{n-1,n-1} is unused.
        X(triangular_banded) * R = P->R;
        for (int i = 0; i < n-2; i++)
            bq[i] = X(get_triangular_banded_index)(R, i+1, i+1)/X(get_triangular_banded_index)(R, i, i)*bp[i];
        aq[0] = ap[0] + X(get_triangular_banded_index)(R, 0, 1)/X(get_triangular_banded_index)(R, 0, 0)*bp[0];
        for (int i = 1; i < n-1; i++)
            aq[i] = (X(get_triangular_banded_index)(R, i, i)*ap[i] + X(get_triangular_banded_index)(R, i, i+1)*bp[i] - X(get_triangular_banded_index)(R, i-1, i)*bq[i-1])/X(get_triangular_banded_index)(R, i, i);
    }
    else {
        // P Lᵀ = Q K => XQ = R XP R⁻¹, where R = K L⁻ᵀ, but we can calculate it only up to n-1.
        X(triangular_banded) * K = P->K;
        X(triangular_banded) * Lt = P->R;
        FLT Rip1ip1 = X(get_triangular_banded_index)(K, 0, 0)/X(get_triangular_banded_index)(Lt, 0, 0);
        for (int i = 0; i < n-2; i++) {
            FLT Rii = Rip1ip1;
            Rip1ip1 = X(get_triangular_banded_index)(K, i+1, i+1)/X(get_triangular_banded_index)(Lt, i+1, i+1);
            bq[i] = Rip1ip1/Rii*bp[i];
        }
        FLT Rii = X(get_triangular_banded_index)(K, 0, 0)/X(get_triangular_banded_index)(Lt, 0, 0);
        FLT Riip1 = (X(get_triangular_banded_index)(K, 0, 1) - Rii*X(get_triangular_banded_index)(Lt, 0, 1))/X(get_triangular_banded_index)(Lt, 1, 1);
        aq[0] = ap[0] + Riip1/Rii*bp[0];
        for (int i = 1; i < n-1; i++) {
            FLT Rim1i = Riip1;
            Rii = X(get_triangular_banded_index)(K, i, i)/X(get_triangular_banded_index)(Lt, i, i);
            Riip1 = (X(get_triangular_banded_index)(K, i, i+1) - Rii*X(get_triangular_banded_index)(Lt, i, i+1))/X(get_triangular_banded_index)(Lt, i+1, i+1);
            aq[i] = (Rii*ap[i] + Riip1*bp[i] - Rim1i*bq[i-1])/Rii;
        }
    }
}

X(symmetric_tridiagonal) * X(execute_jacobi_similarity)(const X(modified_plan) * P, const X(symmetric_tridiagonal) * XP) {
    int n = MIN(XP->n, P->n);
    X(symmetric_tridiagonal) * XQ = malloc(sizeof(X(symmetric_tridiagonal)));
    XQ->a = malloc((n-1)*sizeof(FLT));
    XQ->b = malloc((n-2)*sizeof(FLT));
    XQ->n = n-1;
    Y(execute_jacobi_similarity)(P, n, XP->a, XP->b, XQ->a, XQ->b);
    return XQ;
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

FLT X(rec_A_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta) {
    if (norm) {
        if (n == 0)
            return (alpha+beta+2)/2*Y(sqrt)((alpha+beta+3)/((alpha+1)*(beta+1)));
        else
            return Y(sqrt)(((2*n+alpha+beta+1)*(2*n+alpha+beta+2)*(2*n+alpha+beta+2)*(2*n+alpha+beta+3))/((n+1)*(n+alpha+1)*(n+beta+1)*(n+alpha+beta+1)))/2;
    }
    else {
        if (n == 0)
            return (alpha+beta+2)/2;
        else
            return ((2*n+alpha+beta+1)*(2*n+alpha+beta+2))/(2*(n+1)*(n+alpha+beta+1));
    }
}

FLT X(rec_B_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta) {
    if (norm) {
        if (n == 0)
            return (alpha-beta)/2*Y(sqrt)((alpha+beta+3)/((alpha+1)*(beta+1)));
        else
            return ((alpha-beta)*(alpha+beta))/(2*(2*n+alpha+beta))*Y(sqrt)(((2*n+alpha+beta+1)*(2*n+alpha+beta+3))/((n+1)*(n+alpha+1)*(n+beta+1)*(n+alpha+beta+1)));
    }
    else {
        if (n == 0)
            return (alpha-beta)/2;
        else
            return ((alpha-beta)*(alpha+beta)*(2*n+alpha+beta+1))/(2*(n+1)*(n+alpha+beta+1)*(2*n+alpha+beta));
    }
}

FLT X(rec_C_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta) {
    if (norm) {
        if (n == 1)
            return (alpha+beta+4)/(alpha+beta+2)*Y(sqrt)(((alpha+1)*(beta+1)*(alpha+beta+5))/(2*(alpha+2)*(beta+2)*(alpha+beta+2)));
        else
            return (2*n+alpha+beta+2)/(2*n+alpha+beta)*Y(sqrt)((n*(n+alpha)*(n+beta)*(n+alpha+beta))/((n+1)*(n+alpha+1)*(n+beta+1)*(n+alpha+beta+1))*(2*n+alpha+beta+3)/(2*n+alpha+beta-1));
    }
    else {
        return ((n+alpha)*(n+beta)*(2*n+alpha+beta+2))/((n+1)*(n+alpha+beta+1)*(2*n+alpha+beta));
    }
}

FLT X(rec_A_laguerre)(const int norm, const int n, const FLT alpha) {
    if (norm) {
        return -1/Y(sqrt)((n+1)*(n+alpha+1));
    }
    else {
        return -ONE(FLT)/(n+1);
    }
}

FLT X(rec_B_laguerre)(const int norm, const int n, const FLT alpha) {
    if (norm) {
        return (2*n+alpha+1)/Y(sqrt)((n+1)*(n+alpha+1));
    }
    else {
        return (2*n+alpha+1)/(n+1);
    }
}

FLT X(rec_C_laguerre)(const int norm, const int n, const FLT alpha) {
    if (norm) {
        return Y(sqrt)((n*(n+alpha))/((n+1)*(n+alpha+1)));
    }
    else {
        return (n+alpha)/(n+1);
    }
}

FLT X(rec_A_hermite)(const int norm, const int n) {
    if (norm) {
        return Y(sqrt)(2/(n+ONE(FLT)));
    }
    else {
        return 2;
    }
}

FLT X(rec_B_hermite)(const int norm, const int n) {return 0;}

FLT X(rec_C_hermite)(const int norm, const int n) {
    if (norm) {
        return Y(sqrt)(n/(n+ONE(FLT)));
    }
    else {
        return 2*n;
    }
}

// Dᵏ P^{(α,β)}
X(banded) * X(create_jacobi_derivative)(const int norm, const int m, const int n, const int order, const FLT alpha, const FLT beta) {
    X(banded) * A = X(malloc_banded)(m, n, -order, order);
    if (norm) {
        for (int j = order; j < n; j++) {
            FLT v = 1;
            for (int k = 0; k < order; k++)
                v *= (j-k)*(j+alpha+beta+k+1);
            v = Y(sqrt)(v);
            X(set_banded_index)(A, v, j-order, j);
        }
    }
    else {
        for (int j = order; j < n; j++) {
            FLT v = 1;
            for (int k = 0; k < order; k++)
                v *= (j+alpha+beta+k+1)/2;
            X(set_banded_index)(A, v, j-order, j);
        }
    }
    return A;
}

// x P^{(α,β)}
X(banded) * X(create_jacobi_multiplication)(const int norm, const int m, const int n, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(m, n, 1, 1);
    if (norm) {
        for (int j = 0; j < n; j++) {
            FLT v;
            if (j == 1)
                v = 2*Y(sqrt)(((alpha+1)*(beta+1))/((alpha+beta+2)*(alpha+beta+2)*(alpha+beta+3)));
            else
                v = 2*Y(sqrt)((j*(j+alpha)*(j+beta)*(j+alpha+beta))/((2*j+alpha+beta-1)*(2*j+alpha+beta)*(2*j+alpha+beta)*(2*j+alpha+beta+1)));
            X(set_banded_index)(A, v, j-1, j);
            if (j == 0) {
                v = (beta-alpha)/(alpha+beta+2);
                X(set_banded_index)(A, v, 0, 0);
                v = 2*Y(sqrt)(((alpha+1)*(beta+1))/((alpha+beta+2)*(alpha+beta+2)*(alpha+beta+3)));
                X(set_banded_index)(A, v, 1, 0);
            }
            else {
                v = ((beta-alpha)*(alpha+beta))/((2*j+alpha+beta)*(2*j+alpha+beta+2));
                X(set_banded_index)(A, v, j, j);
                v = 2*Y(sqrt)(((j+1)*(j+alpha+1)*(j+beta+1)*(j+alpha+beta+1))/((2*j+alpha+beta+1)*(2*j+alpha+beta+2)*(2*j+alpha+beta+2)*(2*j+alpha+beta+3)));
                X(set_banded_index)(A, v, j+1, j);
            }
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            FLT v = (2*(j+alpha)*(j+beta))/((2*j+alpha+beta)*(2*j+alpha+beta+1));
            X(set_banded_index)(A, v, j-1, j);
            if (j == 0) {
                v = (beta-alpha)/(alpha+beta+2);
                X(set_banded_index)(A, v, 0, 0);
                v = 2/(alpha+beta+2);
                X(set_banded_index)(A, v, 1, 0);
            }
            else {
                v = ((beta-alpha)*(alpha+beta))/((2*j+alpha+beta)*(2*j+alpha+beta+2));
                X(set_banded_index)(A, v, j, j);
                v = (2*(j+1)*(j+alpha+beta+1))/((2*j+alpha+beta+1)*(2*j+alpha+beta+2));
                X(set_banded_index)(A, v, j+1, j);
            }
        }
    }
    return A;
}

// P^{(α,β)} ↗ P^{(α+1,β+1)}
X(banded) * X(create_jacobi_raising)(const int norm, const int m, const int n, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(m, n, 0, 2);
    if (norm) {
        for (int j = 0; j < n; j++) {
            FLT v = -2*Y(sqrt)(((j-1)*j*(j+alpha)*(j+beta))/((2*j+alpha+beta-1)*(2*j+alpha+beta)*(2*j+alpha+beta)*(2*j+alpha+beta+1)));
            X(set_banded_index)(A, v, j-2, j);
            v = 2*(alpha-beta)*Y(sqrt)(j*(j+alpha+beta+1))/((2*j+alpha+beta)*(2*j+alpha+beta+2));
            X(set_banded_index)(A, v, j-1, j);
            if (j == 0)
                v = 2*Y(sqrt)(((alpha+1)*(beta+1))/((alpha+beta+2)*(alpha+beta+3)));
            else
                v = 2*Y(sqrt)(((j+alpha+1)*(j+beta+1)*(j+alpha+beta+1)*(j+alpha+beta+2))/((2*j+alpha+beta+1)*(2*j+alpha+beta+2)*(2*j+alpha+beta+2)*(2*j+alpha+beta+3)));
            X(set_banded_index)(A, v, j, j);
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            FLT v = -((j+alpha)*(j+beta))/((2*j+alpha+beta)*(2*j+alpha+beta+1));
            X(set_banded_index)(A, v, j-2, j);
            v = ((alpha-beta)*(j+alpha+beta+1))/((2*j+alpha+beta)*(2*j+alpha+beta+2));
            X(set_banded_index)(A, v, j-1, j);
            if (j == 0)
                v = 1;
            else
                v = ((j+alpha+beta+1)*(j+alpha+beta+2))/((2*j+alpha+beta+1)*(2*j+alpha+beta+2));
            X(set_banded_index)(A, v, j, j);
        }
    }
    return A;
}

// (1-x²) P^{(α+1,β+1)} ↘ P^{(α,β)}
X(banded) * X(create_jacobi_lowering)(const int norm, const int m, const int n, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(m, n, 2, 0);
    if (norm) {
        for (int j = 0; j < n; j++) {
            FLT v;
            if (j == 0)
                v = 2*Y(sqrt)(((alpha+1)*(beta+1))/((alpha+beta+2)*(alpha+beta+3)));
            else
                v = 2*Y(sqrt)(((j+alpha+1)*(j+beta+1)*(j+alpha+beta+1)*(j+alpha+beta+2))/((2*j+alpha+beta+1)*(2*j+alpha+beta+2)*(2*j+alpha+beta+2)*(2*j+alpha+beta+3)));
            X(set_banded_index)(A, v, j, j);
            v = 2*(alpha-beta)*Y(sqrt)((j+1)*(j+alpha+beta+2))/((2*j+alpha+beta+2)*(2*j+alpha+beta+4));
            X(set_banded_index)(A, v, j+1, j);
            v = -2*Y(sqrt)(((j+1)*(j+2)*(j+alpha+2)*(j+beta+2))/((2*j+alpha+beta+3)*(2*j+alpha+beta+4)*(2*j+alpha+beta+4)*(2*j+alpha+beta+5)));
            X(set_banded_index)(A, v, j+2, j);
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            FLT v = (4*(j+alpha+1)*(j+beta+1))/((2*j+alpha+beta+2)*(2*j+alpha+beta+3));
            X(set_banded_index)(A, v, j, j);
            v = (4*(alpha-beta)*(j+1))/((2*j+alpha+beta+2)*(2*j+alpha+beta+4));
            X(set_banded_index)(A, v, j+1, j);
            v = -(4*(j+1)*(j+2))/((2*j+alpha+beta+3)*(2*j+alpha+beta+4));
            X(set_banded_index)(A, v, j+2, j);
        }
    }
    return A;
}

// Dᵏ L^{(α)}
X(banded) * X(create_laguerre_derivative)(const int norm, const int m, const int n, const int order, const FLT alpha) {
    X(banded) * A = X(malloc_banded)(m, n, -order, order);
    if (norm) {
        for (int j = order; j < n; j++) {
            FLT v = 1;
            for (int k = 0; k < order; k++)
                v *= j-k;
            v = Y(sqrt)(v);
            if (order%2) v = -v;
            X(set_banded_index)(A, v, j-order, j);
        }
    }
    else {
        for (int j = order; j < n; j++) {
            FLT v = 1;
            if (order%2) v = -v;
            X(set_banded_index)(A, v, j-order, j);
        }
    }
    return A;
}

// x L^{(α)}
X(banded) * X(create_laguerre_multiplication)(const int norm, const int m, const int n, const FLT alpha) {
    X(banded) * A = X(calloc_banded)(m, n, 1, 1);
    if (norm) {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, -Y(sqrt)(j*(j+alpha)), j-1, j);
            X(set_banded_index)(A, 2*j+alpha+1, j, j);
            X(set_banded_index)(A, -Y(sqrt)((j+1)*(j+alpha+1)), j+1, j);
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, -(j+alpha), j-1, j);
            X(set_banded_index)(A, 2*j+alpha+1, j, j);
            X(set_banded_index)(A, -(j+1), j+1, j);
        }
    }
    return A;
}

// L^{(α)} ↗ L^{(α+1)}
X(banded) * X(create_laguerre_raising)(const int norm, const int m, const int n, const FLT alpha) {
    X(banded) * A = X(calloc_banded)(m, n, 0, 1);
    if (norm) {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, -Y(sqrt)(j), j-1, j);
            X(set_banded_index)(A, Y(sqrt)(j+alpha+1), j, j);
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, -1, j-1, j);
            X(set_banded_index)(A, 1, j, j);
        }
    }
    return A;
}

// x L^{(α+1)} ↘ L^{(α)}
X(banded) * X(create_laguerre_lowering)(const int norm, const int m, const int n, const FLT alpha) {
    X(banded) * A = X(calloc_banded)(m, n, 1, 0);
    if (norm) {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, Y(sqrt)(j+alpha+1), j, j);
            X(set_banded_index)(A, -Y(sqrt)(j+1), j+1, j);
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, j+alpha+1, j, j);
            X(set_banded_index)(A, -(j+1), j+1, j);
        }
    }
    return A;
}

// Dᵏ H
X(banded) * X(create_hermite_derivative)(const int norm, const int m, const int n, const int order) {
    X(banded) * A = X(malloc_banded)(m, n, -order, order);
    if (norm) {
        for (int j = order; j < n; j++) {
            FLT v = 1;
            for (int k = 0; k < order; k++)
                v *= 2*(j-k);
            v = Y(sqrt)(v);
            X(set_banded_index)(A, v, j-order, j);
        }
    }
    else {
        for (int j = order; j < n; j++) {
            FLT v = 1;
            for (int k = 0; k < order; k++)
                v *= 2*(j-k);
            X(set_banded_index)(A, v, j-order, j);
        }
    }
    return A;
}

// x H
X(banded) * X(create_hermite_multiplication)(const int norm, const int m, const int n) {
    X(banded) * A = X(calloc_banded)(m, n, 1, 1);
    if (norm) {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, Y(sqrt)(j/TWO(FLT)), j-1, j);
            X(set_banded_index)(A, Y(sqrt)((j+1)/TWO(FLT)), j+1, j);
        }
    }
    else {
        for (int j = 0; j < n; j++) {
            X(set_banded_index)(A, j, j-1, j);
            X(set_banded_index)(A, 0.5, j+1, j);
        }
    }
    return A;
}

void X(create_legendre_to_chebyshev_diagonal_connection_coefficient)(const int normleg, const int normcheb, const int n, FLT * D, const int INCD) {
    if (normleg) {
        if (normcheb) {
            if (n > 0)
                D[0] = Y(sqrt)(0.5)*Y(tgamma)(0.5);
            if (n > 1)
                D[INCD] = Y(sqrt)(1.5)*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = Y(sqrt)((2*i+1)*(2*i-ONE(FLT)))*D[(i-1)*INCD]/(2*i);
        }
        else {
            if (n > 0)
                D[0] = Y(sqrt)(0.5);
            if (n > 1)
                D[INCD] = Y(sqrt)(1.5);
            for (int i = 2; i < n; i++)
                D[i*INCD] = Y(sqrt)((2*i+1)*(2*i-ONE(FLT)))*D[(i-1)*INCD]/(2*i);
        }
    }
    else {
        if (normcheb) {
            if (n > 0)
                D[0] = Y(tgamma)(0.5);
            if (n > 1)
                D[INCD] = D[0]/Y(sqrt)(2);
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i-1)*D[(i-1)*INCD]/(2*i);
        }
        else {
            if (n > 0)
                D[0] = 1;
            if (n > 1)
                D[INCD] = 1;
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i-1)*D[(i-1)*INCD]/(2*i);
        }
    }
}

void X(create_chebyshev_to_legendre_diagonal_connection_coefficient)(const int normcheb, const int normleg, const int n, FLT * D, const int INCD) {
    if (normcheb) {
        if (normleg) {
            if (n > 0)
                D[0] = Y(sqrt)(2)/Y(tgamma)(0.5);
            if (n > 1)
                D[INCD] = D[0]/Y(sqrt)(1.5);
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i)/Y(sqrt)((2*i+1)*(2*i-ONE(FLT)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1/Y(tgamma)(0.5);
            if (n > 1)
                D[INCD] = Y(sqrt)(2)*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i)*D[(i-1)*INCD]/(2*i-1);
        }
    }
    else {
        if (normleg) {
            if (n > 0)
                D[0] = Y(sqrt)(2);
            if (n > 1)
                D[INCD] = 1/Y(sqrt)(1.5);
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i)/Y(sqrt)((2*i+1)*(2*i-ONE(FLT)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1;
            if (n > 1)
                D[INCD] = 1;
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i)*D[(i-1)*INCD]/(2*i-1);
        }
    }
}

void X(create_ultraspherical_to_ultraspherical_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT lambda, const FLT mu, FLT * D, const int INCD) {
    if (norm1) {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(lambda+1)/(Y(tgamma)(0.5)*Y(tgamma)(lambda+0.5))) * Y(sqrt)(Y(tgamma)(0.5)*Y(tgamma)(mu+0.5)/Y(tgamma)(mu+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)((i-1+mu)/i*(i-1+2*mu)/(i+mu)) * Y(sqrt)(i/(i-1+lambda)*(i+lambda)/(i-1+2*lambda)) * (i-1+lambda)/(i-1+mu)*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(lambda+1)/(Y(tgamma)(0.5)*Y(tgamma)(lambda+0.5)));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)(i/(i-1+lambda)*(i+lambda)/(i-1+2*lambda)) * (i-1+lambda)/(i-1+mu)*D[(i-1)*INCD];
        }
    }
    else {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(0.5)*Y(tgamma)(mu+0.5)/Y(tgamma)(mu+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)((i-1+mu)/i*(i-1+2*mu)/(i+mu)) * (i-1+lambda)/(i-1+mu)*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1;
            for (int i = 1; i < n; i++)
                D[i*INCD] = (i-1+lambda)/(i-1+mu)*D[(i-1)*INCD];
        }
    }
}

void X(create_jacobi_to_jacobi_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta, FLT * D, const int INCD) {
    if (norm1) {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(alpha+beta+2)/(Y(pow)(2, alpha+beta+1)*Y(tgamma)(alpha+1)*Y(tgamma)(beta+1))*(Y(pow)(2, gamma+delta+1)*Y(tgamma)(gamma+1)*Y(tgamma)(delta+1))/Y(tgamma)(gamma+delta+2));
            if (n > 1)
                D[INCD] = (alpha+beta+2)/(gamma+delta+2)*Y(sqrt)((alpha+beta+3)/((alpha+1)*(beta+1))*((gamma+1)*(delta+1))/(gamma+delta+3))*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = (2*i+alpha+beta)/(2*i+gamma+delta)*Y(sqrt)(((2*i+alpha+beta-1)*(2*i+alpha+beta+1))/((i+alpha)*(i+beta)*(i+alpha+beta))*((i+gamma)*(i+delta)*(i+gamma+delta))/((2*i+gamma+delta-1)*(2*i+gamma+delta+1)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(alpha+beta+2)/(Y(pow)(2, alpha+beta+1)*Y(tgamma)(alpha+1)*Y(tgamma)(beta+1)));
            if (n > 1)
                D[INCD] = (alpha+beta+2)/(gamma+delta+2)*Y(sqrt)((alpha+beta+3)/((alpha+1)*(beta+1)))*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = Y(sqrt)(((2*i+alpha+beta-1)*(2*i+alpha+beta)*(2*i+alpha+beta)*(2*i+alpha+beta+1))/(i*(i+alpha)*(i+beta)*(i+alpha+beta)))*(i*(i+gamma+delta))/((2*i+gamma+delta-1)*(2*i+gamma+delta))*D[(i-1)*INCD];
        }
    }
    else {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)((Y(pow)(2, gamma+delta+1)*Y(tgamma)(gamma+1)*Y(tgamma)(delta+1))/Y(tgamma)(gamma+delta+2));
            if (n > 1)
                D[INCD] = (alpha+beta+2)/(gamma+delta+2)*Y(sqrt)(((gamma+1)*(delta+1))/(gamma+delta+3))*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = ((2*i+alpha+beta-1)*(2*i+alpha+beta))/(i*(i+alpha+beta))*Y(sqrt)((i*(i+gamma)*(i+delta)*(i+gamma+delta))/((2*i+gamma+delta-1)*(2*i+gamma+delta)*(2*i+gamma+delta)*(2*i+gamma+delta+1)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1;
            if (n > 1)
                D[INCD] = (alpha+beta+2)/(gamma+delta+2);
            for (int i = 2; i < n; i++)
                D[i*INCD] = ((2*i+alpha+beta-1)*(2*i+alpha+beta)*(i+gamma+delta))/((i+alpha+beta)*(2*i+gamma+delta-1)*(2*i+gamma+delta))*D[(i-1)*INCD];
        }
    }
}

void X(create_laguerre_to_laguerre_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT alpha, const FLT beta, FLT * D, const int INCD) {
    if (norm1) {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(beta+1)/Y(tgamma)(alpha+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)((i+beta)/(i+alpha))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1/Y(sqrt)(Y(tgamma)(alpha+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)(i/(i+alpha))*D[(i-1)*INCD];
        }
    }
    else {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(beta+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)((i+beta)/i)*D[(i-1)*INCD];
        }
        else {
            for (int i = 0; i < n; i++)
                D[i*INCD] = 1;
        }
    }
}

void X(create_associated_jacobi_to_jacobi_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT c, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta, FLT * D, const int INCD) {
    if (norm1) {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(((2*c+alpha+beta+1)*Y(tgamma)(c+alpha+beta+1)*Y(tgamma)(c+1))/(Y(pow)(2, alpha+beta+1)*Y(tgamma)(c+alpha+1)*Y(tgamma)(c+beta+1))*(Y(pow)(2, gamma+delta+1)*Y(tgamma)(gamma+1)*Y(tgamma)(delta+1))/Y(tgamma)(gamma+delta+2));
            if (n > 1)
                D[INCD] = Y(sqrt)(((2*c+alpha+beta+1)*(2*c+alpha+beta+2)*(2*c+alpha+beta+2)*(2*c+alpha+beta+3))/((c+1)*(c+alpha+1)*(c+beta+1)*(c+alpha+beta+1))*((gamma+1)*(delta+1))/((gamma+delta+2)*(gamma+delta+2)*(gamma+delta+3)))*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = Y(sqrt)(((2*(i+c)+alpha+beta-1)*(2*(i+c)+alpha+beta)*(2*(i+c)+alpha+beta)*(2*(i+c)+alpha+beta+1))/((i+c)*(i+c+alpha)*(i+c+beta)*(i+c+alpha+beta))*(i*(i+gamma)*(i+delta)*(i+gamma+delta))/((2*i+gamma+delta-1)*(2*i+gamma+delta)*(2*i+gamma+delta)*(2*i+gamma+delta+1)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = Y(sqrt)(((2*c+alpha+beta+1)*Y(tgamma)(c+alpha+beta+1)*Y(tgamma)(c+1))/(Y(pow)(2, alpha+beta+1)*Y(tgamma)(c+alpha+1)*Y(tgamma)(c+beta+1)));
            if (n > 1)
                D[INCD] = Y(sqrt)(((2*c+alpha+beta+1)*(2*c+alpha+beta+2)*(2*c+alpha+beta+2)*(2*c+alpha+beta+3))/((c+1)*(c+alpha+1)*(c+beta+1)*(c+alpha+beta+1)))/(gamma+delta+2)*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = Y(sqrt)(((2*(i+c)+alpha+beta-1)*(2*(i+c)+alpha+beta)*(2*(i+c)+alpha+beta)*(2*(i+c)+alpha+beta+1))/((i+c)*(i+c+alpha)*(i+c+beta)*(i+c+alpha+beta)))*(i*(i+gamma+delta))/((2*i+gamma+delta-1)*(2*i+gamma+delta))*D[(i-1)*INCD];
        }
    }
    else {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)((Y(pow)(2, gamma+delta+1)*Y(tgamma)(gamma+1)*Y(tgamma)(delta+1))/Y(tgamma)(gamma+delta+2));
            if (n > 1)
                D[INCD] = ((2*c+alpha+beta+1)*(2*c+alpha+beta+2))/((c+1)*(c+alpha+beta+1))*Y(sqrt)(((gamma+1)*(delta+1))/((gamma+delta+2)*(gamma+delta+2)*(gamma+delta+3)))*D[0];
            for (int i = 2; i < n; i++)
                D[i*INCD] = ((2*(i+c)+alpha+beta-1)*(2*(i+c)+alpha+beta))/((i+c)*(i+c+alpha+beta))*Y(sqrt)((i*(i+gamma)*(i+delta)*(i+gamma+delta))/((2*i+gamma+delta-1)*(2*i+gamma+delta)*(2*i+gamma+delta)*(2*i+gamma+delta+1)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1;
            if (n > 1)
                D[INCD] = ((2*c+alpha+beta+1)*(2*c+alpha+beta+2))/((c+1)*(c+alpha+beta+1)*(gamma+delta+2));
            for (int i = 2; i < n; i++)
                D[i*INCD] = ((2*(i+c)+alpha+beta-1)*(2*(i+c)+alpha+beta)*i*(i+gamma+delta))/((i+c)*(i+c+alpha+beta)*(2*i+gamma+delta-1)*(2*i+gamma+delta))*D[(i-1)*INCD];
        }
    }
}

void X(create_associated_laguerre_to_laguerre_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT c, const FLT alpha, const FLT beta, FLT * D, const int INCD) {
    if (norm1) {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(c+1)*Y(tgamma)(beta+1)/Y(tgamma)(c+alpha+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)(i*(i+beta)/((i+c)*(i+alpha+c)))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(c+1)/Y(tgamma)(c+alpha+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = i*D[(i-1)*INCD]/Y(sqrt)((i+c)*(i+alpha+c));
        }
    }
    else {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(beta+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)(i*(i+beta))/(i+c)*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1;
            for (int i = 1; i < n; i++)
                D[i*INCD] = i/(i+c)*D[(i-1)*INCD];
        }
    }
}

void X(create_associated_hermite_to_hermite_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT c, FLT * D, const int INCD) {
    if (norm1) {
        if (norm2) {
            if (n > 0)
                D[0] = 1/Y(sqrt)(Y(pow)(2, c)*Y(tgamma)(c+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)(i/(i+c))*D[(i-1)*INCD];
        }
        else {
            if (n > 0)
                D[0] = 1/Y(sqrt)(Y(tgamma)(0.5)*Y(pow)(2, c)*Y(tgamma)(c+1));
            for (int i = 1; i < n; i++)
                D[i*INCD] = D[(i-1)*INCD]/Y(sqrt)(2*(i+c));
        }
    }
    else {
        if (norm2) {
            if (n > 0)
                D[0] = Y(sqrt)(Y(tgamma)(0.5));
            for (int i = 1; i < n; i++)
                D[i*INCD] = Y(sqrt)(2*i)*D[(i-1)*INCD];
        }
        else {
            for (int i = 0; i < n; i++)
                D[i*INCD] = 1;
        }
    }
}


X(triangular_banded) * X(create_A_legendre_to_chebyshev)(const int norm, const int n) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (n > 1)
        X(set_triangular_banded_index)(A, 2, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(A, -i*(i-ONE(FLT)), i-2, i);
        X(set_triangular_banded_index)(A, i*(i+ONE(FLT)), i, i);
    }
    return A;
}

X(triangular_banded) * X(create_B_legendre_to_chebyshev)(const int norm, const int n) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (n > 0)
        X(set_triangular_banded_index)(B, norm ? Y(sqrt)(2) : 2, 0, 0);
    if (n > 1)
        X(set_triangular_banded_index)(B, 1, 1, 1);
    for (int i = 2; i < n; i++) {
        X(set_triangular_banded_index)(B, -1, i-2, i);
        X(set_triangular_banded_index)(B, 1, i, i);
    }
    return B;
}

X(triangular_banded) * X(create_A_chebyshev_to_legendre)(const int norm, const int n) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (norm) {
        if (n > 1)
            X(set_triangular_banded_index)(A, Y(sqrt)(TWO(FLT)/5), 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(A, -(i+1)*Y(sqrt)(((i-ONE(FLT))*i)/((2*i-ONE(FLT))*(2*i+1)))*(i+1), i-2, i);
            X(set_triangular_banded_index)(A, i*Y(sqrt)(((i+ONE(FLT))*(i+2))/((2*i+ONE(FLT))*(2*i+3)))*i, i, i);
        }
    }
    else {
        if (n > 1)
            X(set_triangular_banded_index)(A, ONE(FLT)/3, 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(A, -(i+1)/(2*i+ONE(FLT))*(i+1), i-2, i);
            X(set_triangular_banded_index)(A, i/(2*i+ONE(FLT))*i, i, i);
        }
    }
    return A;
}

X(triangular_banded) * X(create_B_chebyshev_to_legendre)(const int norm, const int n) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (norm) {
        if (n > 0)
            X(set_triangular_banded_index)(B, Y(sqrt)(TWO(FLT)/3), 0, 0);
        if (n > 1)
            X(set_triangular_banded_index)(B, Y(sqrt)(TWO(FLT)/5), 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(B, -Y(sqrt)(((i-ONE(FLT))*i)/((2*i-ONE(FLT))*(2*i+1))), i-2, i);
            X(set_triangular_banded_index)(B, Y(sqrt)(((i+ONE(FLT))*(i+2))/((2*i+ONE(FLT))*(2*i+3))), i, i);
        }
    }
    else {
        if (n > 0)
            X(set_triangular_banded_index)(B, 1, 0, 0);
        if (n > 1)
            X(set_triangular_banded_index)(B, ONE(FLT)/3, 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(B, -1/(2*i+ONE(FLT)), i-2, i);
            X(set_triangular_banded_index)(B, 1/(2*i+ONE(FLT)), i, i);
        }
    }
    return B;
}

X(triangular_banded) * X(create_A_ultraspherical_to_ultraspherical)(const int norm, const int n, const FLT lambda, const FLT mu) {
    X(triangular_banded) * A = X(calloc_triangular_banded)(n, 2);
    if (norm) {
        if (n > 1)
            X(set_triangular_banded_index)(A, (1+2*lambda)*Y(copysign)(Y(sqrt)((2*mu+1)/(2*mu+4)), mu), 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(A, -(i+2*mu)*(i+2*(mu-lambda))*Y(copysign)(Y(sqrt)((i*(i-ONE(FLT)))/(4*(i+mu)*(i+mu-1))), mu), i-2, i);
            X(set_triangular_banded_index)(A, i*(i+2*lambda)*Y(copysign)(Y(sqrt)(((i+2*mu)*(i+2*mu+1))/(4*(i+mu)*(i+mu+1))), mu), i, i);
        }
    }
    else {
        if (n > 1)
            X(set_triangular_banded_index)(A, (1+2*lambda)*mu/(1+mu), 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(A, -(i+2*mu)*(i+2*(mu-lambda))*mu/(i+mu), i-2, i);
            X(set_triangular_banded_index)(A, i*(i+2*lambda)*mu/(i+mu), i, i);
        }
    }
    return A;
}

X(triangular_banded) * X(create_B_ultraspherical_to_ultraspherical)(const int norm, const int n, const FLT mu) {
    X(triangular_banded) * B = X(calloc_triangular_banded)(n, 2);
    if (norm) {
        if (n > 0)
            X(set_triangular_banded_index)(B, Y(sqrt)((2*mu+1)/(2*mu+2)), 0, 0);
        if (n > 1)
            X(set_triangular_banded_index)(B, Y(copysign)(Y(sqrt)((2*mu+1)/(2*mu+4)), mu), 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(B, -Y(copysign)(Y(sqrt)((i*(i-ONE(FLT)))/(4*(i+mu)*(i+mu-1))), mu), i-2, i);
            X(set_triangular_banded_index)(B, Y(copysign)(Y(sqrt)(((i+2*mu)*(i+2*mu+1))/(4*(i+mu)*(i+mu+1))), mu), i, i);
        }
    }
    else {
        if (n > 0)
            X(set_triangular_banded_index)(B, 1, 0, 0);
        if (n > 1)
            X(set_triangular_banded_index)(B, mu/(1+mu), 1, 1);
        for (int i = 2; i < n; i++) {
            X(set_triangular_banded_index)(B, -mu/(i+mu), i-2, i);
            X(set_triangular_banded_index)(B, mu/(i+mu), i, i);
        }
    }
    return B;
}

X(triangular_banded) * X(create_A_jacobi_to_jacobi)(const int norm, const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X(banded) * A = X(calloc_banded)(n, n, 0, 2);

    X(banded) * D1 = X(create_jacobi_derivative)(norm, n, n, 1, gamma, delta);
    X(banded) * D2 = X(create_jacobi_derivative)(norm, n, n, 2, gamma, delta);
    X(banded) * L1 = X(create_jacobi_lowering)(norm, n, n, gamma+1, delta+1);
    X(banded) * M1 = X(create_jacobi_multiplication)(norm, n, n, gamma+1, delta+1);

    // A = σ D² + τ D
    X(gbmm)(-1, L1, D2, 0, A);
    X(banded_add)(1, A, alpha-beta, D1, A);
    X(gbmm)(alpha+beta+2, M1, D1, 1, A);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(L1);
    X(destroy_banded)(M1);

    return X(convert_banded_to_triangular_banded)(A);
}

X(triangular_banded) * X(create_B_jacobi_to_jacobi)(const int norm, const int n, const FLT gamma, const FLT delta) {
    X(banded) * B = X(create_jacobi_raising)(norm, n, n, gamma, delta);
    return X(convert_banded_to_triangular_banded)(B);
}

X(triangular_banded) * X(create_A_laguerre_to_laguerre)(const int norm, const int n, const FLT alpha, const FLT beta) {
    X(triangular_banded) * A = X(malloc_triangular_banded)(n, 1);
    if (norm) {
        for (int i = 0; i < n; i++) {
            X(set_triangular_banded_index)(A, (alpha-beta-i)*Y(sqrt)(i), i-1, i);
            X(set_triangular_banded_index)(A, i*Y(sqrt)(i+beta+1), i, i);
        }
    }
    else {
        for (int i = 0; i < n; i++) {
            X(set_triangular_banded_index)(A, alpha-beta-i, i-1, i);
            X(set_triangular_banded_index)(A, i, i, i);
        }
    }
    return A;
}

X(triangular_banded) * X(create_B_laguerre_to_laguerre)(const int norm, const int n, const FLT beta) {
    X(banded) * B = X(create_laguerre_raising)(norm, n, n, beta);
    return X(convert_banded_to_triangular_banded)(B);
}

X(triangular_banded) * X(create_A_associated_jacobi_to_jacobi)(const int norm, const int n, const int c, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta) {
    X(banded) * A = X(calloc_banded)(n, n, 0, 4);

    FLT lambdacm1 = (c+alpha+beta)*(c-1);

    X(banded) * D1 = X(create_jacobi_derivative)(norm, n, n, 1, gamma, delta);
    X(banded) * D2 = X(create_jacobi_derivative)(norm, n, n, 2, gamma, delta);
    X(banded) * D3 = X(create_jacobi_derivative)(norm, n, n, 3, gamma, delta);
    X(banded) * D4 = X(create_jacobi_derivative)(norm, n, n, 4, gamma, delta);
    X(banded) * R0 = X(create_jacobi_raising)(norm, n, n, gamma, delta);
    X(banded) * R1 = X(create_jacobi_raising)(norm, n, n, gamma+1, delta+1);
    X(banded) * L1 = X(create_jacobi_lowering)(norm, n, n, gamma+1, delta+1);
    X(banded) * L2 = X(create_jacobi_lowering)(norm, n, n, gamma+2, delta+2);
    X(banded) * L3 = X(create_jacobi_lowering)(norm, n, n, gamma+3, delta+3);
    X(banded) * M2 = X(create_jacobi_multiplication)(norm, n, n, gamma+2, delta+2);

    // A4 = -σ² D⁴
    //    = -(x²-1)² D⁴
    //    = -L2*L3*D4

    X(banded) * A4a = X(calloc_banded)(n, n, -2, 4);
    X(gbmm)(1, L3, D4, 0, A4a);
    X(banded) * A4 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(-1, L2, A4a, 0, A4);

    // A3 = -5 σ σ' D³
    //    = 10 x (1-x²) D³
    //    = 10*M2*L2*D3

    X(banded) * A3a = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, L2, D3, 0, A3a);
    X(banded) * A3 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(10, M2, A3a, 0, A3);

    // A2 = [ τ²+2τ'σ-2τσ'-6σσ''+4*λ_{c-1}*σ-3*σ'² ] D²
    //    = { 2*(α²+β²)-16 + 2*(α-β)*(α+β)*x + [24-4*λ_{c-1}-(α+β)*(α+β+2)]*(1-x²) } D²
    //    = { 2*(α²+β²)-16 + 2*(α-β)*(α+β)*M2 + [24-4*λ_{c-1}-(α+β)*(α+β+2)]*R1*L1 }*D2

    X(banded) * A2a = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(1, L1, D2, 0, A2a);
    X(banded) * A2b = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, R1, A2a, 0, A2b);
    X(banded) * A2c = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, M2, D2, 0, A2c);
    X(banded) * A2 = X(calloc_banded)(n, n, 0, 4);
    X(banded_add)(24-4*lambdacm1-(alpha+beta)*(alpha+beta+2), A2b, 2*(alpha-beta)*(alpha+beta), A2c, A2);
    X(banded_add)(1, A2, 2*(alpha*alpha+beta*beta)-16, D2, A2);

    // A1 = 3*[ τ*τ'+2*λ_{c-1}*σ'-(τ+σ')*σ'' ] D
    //    = 3*{ (α-β)*(α+β)+[(α+β+1)²+4*λ_{c-1}-5]*x } D
    //    = 3*{ (α-β)*(α+β)+[(α+β+1)²+4*λ_{c-1}-5]*M2 }*R1*D1

    X(banded) * A1a = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, R1, D1, 0, A1a);
    X(banded) * A1b = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, M2, A1a, 0, A1b);
    X(banded) * A1 = X(calloc_banded)(n, n, 0, 4);
    X(banded_add)(3*(alpha-beta)*(alpha+beta), A1a, 3*((alpha+beta+1)*(alpha+beta+1)+4*lambdacm1-5), A1b, A1);

    // A0 = [ 2*λ_{c-1}*σ'' - τ'(σ''-τ') ]*I
    //    = [ 4λ_{c-1} + (α+β)*(α+β+2) ]*R1*R0

    X(banded) * A0 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(4*lambdacm1+(alpha+beta)*(alpha+beta+2), R1, R0, 0, A0);

    // A = A4+A3+A2+A1+A0

    X(banded_add)(1, A0, 1, A1, A);
    X(banded_add)(1, A, 1, A2, A);
    X(banded_add)(1, A, 1, A3, A);
    X(banded_add)(1, A, 1, A4, A);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(D3);
    X(destroy_banded)(D4);
    X(destroy_banded)(R0);
    X(destroy_banded)(R1);
    X(destroy_banded)(L1);
    X(destroy_banded)(L2);
    X(destroy_banded)(L3);
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
    X(destroy_banded)(A1);
    X(destroy_banded)(A0);

    return X(convert_banded_to_triangular_banded)(A);
}

X(triangular_banded) * X(create_B_associated_jacobi_to_jacobi)(const int norm, const int n, const FLT gamma, const FLT delta) {
    X(banded) * B = X(calloc_banded)(n, n, 0, 4);

    X(banded) * D1 = X(create_jacobi_derivative)(norm, n, n, 1, gamma, delta);
    X(banded) * D2 = X(create_jacobi_derivative)(norm, n, n, 2, gamma, delta);
    X(banded) * R0 = X(create_jacobi_raising)(norm, n, n, gamma, delta);
    X(banded) * R1 = X(create_jacobi_raising)(norm, n, n, gamma+1, delta+1);
    X(banded) * L1 = X(create_jacobi_lowering)(norm, n, n, gamma+1, delta+1);
    X(banded) * M2 = X(create_jacobi_multiplication)(norm, n, n, gamma+2, delta+2);

    // B2 = 2(x²-1) D² == -2*R1*L1*D2

    X(banded) * B2a = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(1, L1, D2, 0, B2a);
    X(banded) * B2 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(-2, R1, B2a, 0, B2);

    // B1 = 6 x D == 6*M2*R1*D1

    X(banded) * B1a = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, R1, D1, 0, B1a);
    X(banded) * B1 = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(6, M2, B1a, 0, B1);

    // B = B2+B1+2R1*R0

    X(gbmm)(2, R1, R0, 0, B);
    X(banded_add)(1, B, 1, B1, B);
    X(banded_add)(1, B, 1, B2, B);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(R0);
    X(destroy_banded)(R1);
    X(destroy_banded)(L1);
    X(destroy_banded)(M2);

    X(destroy_banded)(B2a);
    X(destroy_banded)(B2);
    X(destroy_banded)(B1a);
    X(destroy_banded)(B1);

    return X(convert_banded_to_triangular_banded)(B);
}

X(triangular_banded) * X(create_C_associated_jacobi_to_jacobi)(const int norm, const int n, const FLT gamma, const FLT delta) {
    X(banded) * C = X(calloc_banded)(n, n, 0, 4);

    X(banded) * R0 = X(create_jacobi_raising)(norm, n, n, gamma, delta);
    X(banded) * R1 = X(create_jacobi_raising)(norm, n, n, gamma+1, delta+1);

    X(gbmm)(1, R1, R0, 0, C);

    X(destroy_banded)(R0);
    X(destroy_banded)(R1);

    return X(convert_banded_to_triangular_banded)(C);
}

X(triangular_banded) * X(create_A_associated_laguerre_to_laguerre)(const int norm, const int n, const int c, const FLT alpha, const FLT beta) {
    X(banded) * A = X(calloc_banded)(n, n, 0, 4);

    X(banded) * D1 = X(create_laguerre_derivative)(norm, n, n, 1, beta);
    X(banded) * D2 = X(create_laguerre_derivative)(norm, n, n, 2, beta);
    X(banded) * D3 = X(create_laguerre_derivative)(norm, n, n, 3, beta);
    X(banded) * D4 = X(create_laguerre_derivative)(norm, n, n, 4, beta);
    X(banded) * R0 = X(create_laguerre_raising)(norm, n, n, beta);
    X(banded) * R1 = X(create_laguerre_raising)(norm, n, n, beta+1);
    X(banded) * L2 = X(create_laguerre_lowering)(norm, n, n, beta+2);
    X(banded) * L3 = X(create_laguerre_lowering)(norm, n, n, beta+3);
    X(banded) * M2 = X(create_laguerre_multiplication)(norm, n, n, beta+2);

    // A4 = -σ² D⁴
    //    = -(-x)² D⁴
    //    = -L2*L3*D4

    X(banded) * A4a = X(calloc_banded)(n, n, -3, 4);
    X(gbmm)(1, L3, D4, 0, A4a);
    X(banded) * A4 = X(calloc_banded)(n, n, -2, 4);
    X(gbmm)(-1, L2, A4a, 0, A4);

    // A3 = -5 σ σ' D³
    //    = -5x D³
    //    = -5*L2*D3

    X(banded) * A3 = X(calloc_banded)(n, n, -2, 3);
    X(gbmm)(-5, L2, D3, 0, A3);

    // A2 = [ τ²+2τ'σ-2τσ'-6σσ''+4*λ_{c-1}*σ-3*σ'² ] D²
    //    = [ x²-2*(α+2c-1)*x+α²-4 ] D²
    //    = [ M2*M2 - 2*(α+2c-1)*M2 + (α-2)*(α+2) ]*D2

    X(banded) * A2a = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(1, M2, D2, 0, A2a);
    X(banded) * A2b = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, M2, A2a, 0, A2b);
    X(banded) * A2 = X(calloc_banded)(n, n, 0, 4);
    X(banded_add)(-2*(alpha+2*c-1), A2a, 1, A2b, A2);
    X(banded_add)(1, A2, (alpha-2)*(alpha+2), D2, A2);

    // A1 = 3*[ τ*τ'+2*λ_{c-1}*σ'-(τ+σ')*σ'' ] D
    //    = 3*[1-α-2c+x] D
    //    = 3*[1-α-2c+M2]*R1*D1

    X(banded) * A1a = X(calloc_banded)(n, n, -1, 2);
    X(gbmm)(1, R1, D1, 0, A1a);
    X(banded) * A1b = X(calloc_banded)(n, n, 0, 3);
    X(gbmm)(1, M2, A1a, 0, A1b);
    X(banded) * A1 = X(calloc_banded)(n, n, 0, 3);
    X(banded_add)(3*(1-alpha-2*c), A1a, 3, A1b, A1);

    // A0 = [ 2*λ_{c-1}*σ'' - τ'(σ''-τ') ]*I
    //    = R1*R0

    X(banded) * A0 = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(1, R1, R0, 0, A0);

    // A = A4+A3+A2+A1+A0

    X(banded_add)(1, A0, 1, A1, A);
    X(banded_add)(1, A, 1, A2, A);
    X(banded_add)(1, A, 1, A3, A);
    X(banded_add)(1, A, 1, A4, A);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(D3);
    X(destroy_banded)(D4);
    X(destroy_banded)(R0);
    X(destroy_banded)(R1);
    X(destroy_banded)(L2);
    X(destroy_banded)(L3);
    X(destroy_banded)(M2);

    X(destroy_banded)(A4a);
    X(destroy_banded)(A4);
    X(destroy_banded)(A3);
    X(destroy_banded)(A2a);
    X(destroy_banded)(A2b);
    X(destroy_banded)(A2);
    X(destroy_banded)(A1a);
    X(destroy_banded)(A1b);
    X(destroy_banded)(A1);
    X(destroy_banded)(A0);

    return X(convert_banded_to_triangular_banded)(A);
}

X(triangular_banded) * X(create_B_associated_laguerre_to_laguerre)(const int norm, const int n, const FLT beta) {
    X(banded) * B = X(calloc_banded)(n, n, 0, 3);

    X(banded) * D1 = X(create_laguerre_derivative)(norm, n, n, 1, beta);
    X(banded) * D2 = X(create_laguerre_derivative)(norm, n, n, 2, beta);
    X(banded) * R1 = X(create_laguerre_raising)(norm, n, n, beta+1);
    X(banded) * M2 = X(create_laguerre_multiplication)(norm, n, n, beta+2);

    // B2 = -2x D² == -2*M2*D2

    X(banded) * B2 = X(calloc_banded)(n, n, -1, 3);
    X(gbmm)(-2, M2, D2, 0, B2);

    // B1 = -3 D == -3*R1*D1

    X(banded) * B1 = X(calloc_banded)(n, n, -1, 2);
    X(gbmm)(-3, R1, D1, 0, B1);

    // B = B2+B1

    X(banded_add)(1, B1, 1, B2, B);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(R1);
    X(destroy_banded)(M2);

    X(destroy_banded)(B2);
    X(destroy_banded)(B1);

    return X(convert_banded_to_triangular_banded)(B);
}

X(triangular_banded) * X(create_C_associated_laguerre_to_laguerre)(const int norm, const int n, const FLT beta) {
    X(banded) * C = X(calloc_banded)(n, n, 0, 2);

    X(banded) * R0 = X(create_laguerre_raising)(norm, n, n, beta);
    X(banded) * R1 = X(create_laguerre_raising)(norm, n, n, beta+1);

    X(gbmm)(1, R1, R0, 0, C);

    X(destroy_banded)(R0);
    X(destroy_banded)(R1);

    return X(convert_banded_to_triangular_banded)(C);
}

X(triangular_banded) * X(create_A_associated_hermite_to_hermite)(const int norm, const int n, const int c) {
    X(banded) * A = X(calloc_banded)(n, n, 0, 4);

    X(banded) * D1 = X(create_hermite_derivative)(norm, n, n, 1);
    X(banded) * D2 = X(create_hermite_derivative)(norm, n, n, 2);
    X(banded) * D4 = X(create_hermite_derivative)(norm, n, n, 4);
    X(banded) * M1 = X(create_hermite_multiplication)(norm, n, n);
    X(banded) * Ma = X(create_hermite_multiplication)(norm, n, n+1);
    X(banded) * Mb = X(create_hermite_multiplication)(norm, n+1, n);
    X(banded) * M2 = X(calloc_banded)(n, n, 2, 2);
    X(gbmm)(1, Ma, Mb, 0, M2);

    // A4 = -σ² D⁴
    //    = -D4

    X(banded) * A4 = X(calloc_banded)(n, n, -4, 4);
    X(banded_add)(0, A4, -1, D4, A4);

    // A2 = [ τ²+2τ'σ-2τσ'-6σσ''+4*λ_{c-1}*σ-3*σ'² ] D²
    //    = [ 4x² - 4 - 4λ_{c-1} ] D²
    //    = 4*(M2+1-2*c)*D2

    X(banded) * A2a = X(calloc_banded)(n, n, 0, 4);
    X(gbmm)(1, M2, D2, 0, A2a);
    X(banded) * A2 = X(calloc_banded)(n, n, 0, 4);
    X(banded_add)(4, A2a, 4*(1-2*c), D2, A2);

    // A1 = 3*[ τ*τ'+2*λ_{c-1}*σ'-(τ+σ')*σ'' ] D
    //    = 12x D
    //    = 12*M1*D1

    X(banded) * A1 = X(calloc_banded)(n, n, 0, 2);
    X(gbmm)(12, M1, D1, 0, A1);

    // A0 = [ 2*λ_{c-1}*σ'' - τ'(σ''-τ') ]*I
    //    = 4*I

    X(banded) * A0 = X(calloc_banded)(n, n, 0, 0);
    for (int j = 0; j < n; j++)
        X(set_banded_index)(A0, 4, j, j);

    // A = A4+A2+A1+A0

    X(banded_add)(1, A0, 1, A1, A);
    X(banded_add)(1, A, 1, A2, A);
    X(banded_add)(1, A, 1, A4, A);

    X(destroy_banded)(D1);
    X(destroy_banded)(D2);
    X(destroy_banded)(D4);
    X(destroy_banded)(M1);
    X(destroy_banded)(Ma);
    X(destroy_banded)(Mb);
    X(destroy_banded)(M2);

    X(destroy_banded)(A4);
    X(destroy_banded)(A2a);
    X(destroy_banded)(A2);
    X(destroy_banded)(A1);
    X(destroy_banded)(A0);

    return X(convert_banded_to_triangular_banded)(A);
}

X(triangular_banded) * X(create_B_associated_hermite_to_hermite)(const int norm, const int n) {
    X(banded) * B = X(calloc_banded)(n, n, 0, 2);
    X(banded) * D2 = X(create_hermite_derivative)(norm, n, n, 2);

    X(banded_add)(0, B, -2, D2, B);

    return X(convert_banded_to_triangular_banded)(B);
}

X(triangular_banded) * X(create_C_associated_hermite_to_hermite)(const int n) {return X(create_I_triangular_banded)(n, 0);}

X(banded) * X(operator_normalized_jacobi_clenshaw)(const int n, const int nc, const FLT * c, const int incc, const X(cop_params) params) {
    FLT alpha = params.alpha;
    FLT beta = params.beta;
    X(banded) * X = X(create_jacobi_multiplication)(1, n+nc, n+nc, alpha, beta);
    FLT * A = malloc(nc*sizeof(FLT));
    FLT * B = malloc(nc*sizeof(FLT));
    FLT * C = malloc((nc+1)*sizeof(FLT));
    for (int k = 0; k < nc; k++) {
        A[k] = X(rec_A_jacobi)(1, k, alpha, beta);
        B[k] = X(rec_B_jacobi)(1, k, alpha, beta);
        C[k] = X(rec_C_jacobi)(1, k, alpha, beta);
    }
    C[nc] = X(rec_C_jacobi)(1, nc, alpha, beta);
    FLT phi0 = Y(sqrt)(Y(tgamma)(alpha+beta+2)/(Y(pow)(2, alpha+beta+1)*Y(tgamma)(alpha+1)*Y(tgamma)(beta+1)));
    X(banded) * Mlong = X(operator_orthogonal_polynomial_clenshaw)(nc, c, incc, A, B, C, X, phi0);
    X(banded) * M = X(calloc_banded)(n, n, nc-1, nc-1);
    for (int i = 0; i < n*(2*nc-1); i++)
        M->data[i] = Mlong->data[i];
    X(destroy_banded)(X);
    X(destroy_banded)(Mlong);
    free(A);
    free(B);
    free(C);
    return M;
}

X(banded) * X(operator_normalized_laguerre_clenshaw)(const int n, const int nc, const FLT * c, const int incc, const X(cop_params) params) {
    FLT alpha = params.alpha;
    X(banded) * X = X(create_laguerre_multiplication)(1, n+nc, n+nc, alpha);
    FLT * A = malloc(nc*sizeof(FLT));
    FLT * B = malloc(nc*sizeof(FLT));
    FLT * C = malloc((nc+1)*sizeof(FLT));
    for (int k = 0; k < nc; k++) {
        A[k] = X(rec_A_laguerre)(1, k, alpha);
        B[k] = X(rec_B_laguerre)(1, k, alpha);
        C[k] = X(rec_C_laguerre)(1, k, alpha);
    }
    C[nc] = X(rec_C_laguerre)(1, nc, alpha);
    FLT phi0 = Y(sqrt)(1/Y(tgamma)(alpha+1));
    X(banded) * Mlong = X(operator_orthogonal_polynomial_clenshaw)(nc, c, incc, A, B, C, X, phi0);
    X(banded) * M = X(calloc_banded)(n, n, nc-1, nc-1);
    for (int i = 0; i < n*(2*nc-1); i++)
        M->data[i] = Mlong->data[i];
    X(destroy_banded)(X);
    X(destroy_banded)(Mlong);
    free(A);
    free(B);
    free(C);
    return M;
}

X(banded) * X(operator_normalized_hermite_clenshaw)(const int n, const int nc, const FLT * c, const int incc, const X(cop_params) params) {
    X(banded) * X = X(create_hermite_multiplication)(1, n+nc, n+nc);
    FLT * A = malloc(nc*sizeof(FLT));
    FLT * B = malloc(nc*sizeof(FLT));
    FLT * C = malloc((nc+1)*sizeof(FLT));
    for (int k = 0; k < nc; k++) {
        A[k] = X(rec_A_hermite)(1, k);
        B[k] = X(rec_B_hermite)(1, k);
        C[k] = X(rec_C_hermite)(1, k);
    }
    C[nc] = X(rec_C_hermite)(1, nc);
    FLT phi0 = Y(sqrt)(1/Y(tgamma)(0.5));
    X(banded) * Mlong = X(operator_orthogonal_polynomial_clenshaw)(nc, c, incc, A, B, C, X, phi0);
    X(banded) * M = X(calloc_banded)(n, n, nc-1, nc-1);
    for (int i = 0; i < n*(2*nc-1); i++)
        M->data[i] = Mlong->data[i];
    X(destroy_banded)(X);
    X(destroy_banded)(Mlong);
    free(A);
    free(B);
    free(C);
    return M;
}



// a ≤ σ(A) ≤ b and c ≤ σ(B) ≤ d.
static inline void X(compute_spectral_enclosing_sets)(const int m, const FLT * A, const int n, const FLT * B, FLT * a, FLT * b, FLT * c, FLT * d) {
    if (m > 0)
        a[0] = b[0] = A[0];
    if (n > 0)
        c[0] = d[0] = B[0];
    for (int i = 0; i < m; i++) {
        if (A[i] < a[0]) a[0] = A[i];
        if (A[i] > b[0]) b[0] = A[i];
    }
    for (int i = 0; i < n; i++) {
        if (B[i] < c[0]) c[0] = B[i];
        if (B[i] > d[0]) d[0] = B[i];
    }
}

static inline int X(adi_number_of_shifts)(const FLT gamma, const FLT epsilon) {
    return (int) (log(16*gamma)*log(4/epsilon)/(M_PI*M_PI)+1);
}

static inline FLT X(det_3x3)(const FLT M[9]) {
    return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]);
}

static inline void X(adi_compute_shifts)(const FLT a, const FLT b, const FLT c, const FLT d, const FLT gamma, const int J, FLT * p, FLT * q) {
    FLT alpha = -1 + 2*(gamma + Y(sqrt)(gamma*(gamma-1)));
    FLT kappa = Y(sqrt)((1-1/alpha)*(1+1/alpha));
    if (kappa != 1) {
        FLT K = X(complete_elliptic_integral)('1', kappa);
        for (int j = 0; j < J; j++) {
            FLT dn;
            X(jacobian_elliptic_functions)((2*j+1)*K/(2*J), kappa, NULL, NULL, &dn, FT_DN);
            p[j] = -alpha*dn;
        }
    }
    else {
        FLT m1 = 1/(4*alpha*alpha);
        FLT K = Y(log)(4*alpha) + m1*(-1+Y(log)(4*alpha));
        for (int j = 0; j < J; j++) {
            FLT u = (2*j+1)*K/(2*J);
            FLT dn = (1 + m1*(Y(sinh)(u)*Y(cosh)(u)+u)*Y(tanh)(u))/Y(cosh)(u);
            p[j] = -alpha*dn;
        }
    }
    FLT MA[9] = {-a*alpha, -b, c, a, b, c, 1, 1, 1};
    FLT A = X(det_3x3)(MA);
    FLT MB[9] = {-a*alpha, -b, c, -alpha, -1, 1, a, b, c};
    FLT B = X(det_3x3)(MB);
    FLT MC[9] = {-alpha, -1, 1, a, b, c, 1, 1, 1};
    FLT C = X(det_3x3)(MC);
    FLT MD[9] = {-a*alpha, -b, c, -alpha, -1, 1, 1, 1, 1};
    FLT D = X(det_3x3)(MD);
    for (int j = 0; j < J; j++) {
        q[j] = (B-A*p[j])/(D-C*p[j]);
        p[j] = (A*p[j]+B)/(C*p[j]+D);
    }
}

/*
static inline X(lowrankmatrix) * X(ddfadi_computational)(const int m, const FLT * A, const int n, const FLT * B, const int b, const FLT * X, const FLT * Y, const int J, const FLT * p, const FLT * q) {
    int r = b*J;
    X(lowrankmatrix) * Z = X(malloc_lowrankmatrix)('3', m, n, r);
    FLT * U = Z->U;
    FLT * S = Z->S;
    FLT * V = Z->V;
    for (int k = 0; k < b; k++)
        for (int i = 0; i < m; i++)
            U[i+k*m] = X[i+k*m]/(A[i]-q[0]);
    for (int j = 1; j < J; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < m; i++)
                U[i+(k+j*b)*m] = U[i+(k+(j-1)*b)*m] + (q[j]-p[j-1])*U[i+(k+(j-1)*b)*m]/(A[i]-q[j]);
    for (int k = 0; k < r*r; k++)
        S[k] = 0;
    for (int j = 0; j < J; j++)
        for (int k = 0; k < b; k++)
            S[(k+j*b)*(r+1)] = q[j]-p[j];
    for (int k = 0; k < b; k++)
        for (int i = 0; i < n; i++)
            V[i+k*n] = Y[i+k*n]/(B[i]-p[0]);
    for (int j = 1; j < J; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < n; i++)
                V[i+(k+j*b)*n] = V[i+(k+(j-1)*b)*n] + (p[j]-q[j-1])*V[i+(k+(j-1)*b)*n]/(B[i]-p[j]);
    return Z;
}
*/

static inline X(lowrankmatrix) * X(ddfadi_computational)(const int m, const FLT * A, const int n, const FLT * B, const int b, const FLT * X, const FLT * Y, const int J, const FLT * p, const FLT * q) {
    int r = b*J;
    X(lowrankmatrix) * Z = X(malloc_lowrankmatrix)('2', m, n, r);
    FLT * U = Z->U;
    FLT * V = Z->V;
    for (int k = 0; k < b; k++)
        for (int i = 0; i < m; i++)
            U[i+k*m] = X[i+k*m]/(A[i]-q[0]);
    for (int j = 1; j < J; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < m; i++)
                U[i+(k+j*b)*m] = U[i+(k+(j-1)*b)*m] + (q[j]-p[j-1])*U[i+(k+(j-1)*b)*m]/(A[i]-q[j]);
    for (int j = 0; j < J; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < m; i++)
                U[i+(k+j*b)*m] *= q[j]-p[j];
    for (int k = 0; k < b; k++)
        for (int i = 0; i < n; i++)
            V[i+k*n] = Y[i+k*n]/(B[i]-p[0]);
    for (int j = 1; j < J; j++)
        for (int k = 0; k < b; k++)
            for (int i = 0; i < n; i++)
                V[i+(k+j*b)*n] = V[i+(k+(j-1)*b)*n] + (p[j]-q[j-1])*V[i+(k+(j-1)*b)*n]/(B[i]-p[j]);
    return Z;
}

/*
Assumptions to solve AZ-ZB = XYᵀ.
 - A is a diagonal m × m matrix,
 - B is a diagonal n × n matrix,
 - rank(XYᵀ) ≤ b,
 - we use J shifts of factored ADI stored in p and q.
*/
X(lowrankmatrix) * X(ddfadi)(const int m, const FLT * A, const int n, const FLT * B, const int r, const FLT * X, const FLT * Y) {
    // Compute sets enclosing the spectra of diagonal A and B.
    FLT a, b, c, d;
    X(compute_spectral_enclosing_sets)(m, A, n, B, &a, &b, &c, &d);
    // Compute shifts
    FLT gamma = Y(fabs)((c-a)*(d-b)/((c-b)*(d-a)));
    FLT epsilon = Y(eps)()/Y(log2)(((FLT) n)/(TB_EIGEN_BLOCKSIZE>>2));
    int J = X(adi_number_of_shifts)(gamma, epsilon);
    FLT * p = malloc(J*sizeof(FLT));
    FLT * q = malloc(J*sizeof(FLT));
    X(adi_compute_shifts)(a, b, c, d, gamma, J, p, q);
    // Call computational routine
    X(lowrankmatrix) * Z = X(ddfadi_computational)(m, A, n, B, r, X, Y, J, p, q);
    free(p);
    free(q);
    return Z;
}
