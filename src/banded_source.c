void X(destroy_banded)(X(banded) * A) {
    free(A->data);
    free(A);
}

void X(destroy_triangular_banded)(X(triangular_banded) * A) {
    free(A->data);
    free(A);
}

void X(destroy_banded_qr)(X(banded_qr) * F) {
    X(destroy_banded)(F->factors);
    free(F->tau);
    free(F);
}

void X(destroy_tb_eigen_FMM)(X(tb_eigen_FMM) * F) {
    if (F->n < TB_EIGEN_BLOCKSIZE) {
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
    FLT ab, c;
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
    X(banded) * R = X(malloc_banded)(m, n, l, l+u);
    FLT * tau = malloc(MIN(m, n)*sizeof(FLT));
    FLT * D = R->data;
    FLT * B = A->data;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < l; i++)
            D[i+j*(2*l+u+1)] = 0;
        for (int i = 0; i < l+u+1; i++)
            D[l+i+j*(2*l+u+1)] = B[i+j*(l+u+1)];
    }
    u = R->u;
    int nu = l+u+1;
    for (int j = 0; j < MIN(m, n); j++) {
        int lh = MIN(l+1, m-j+1);
        FLT * v = D+u+j*nu;
        tau[j] = X(compute_reflector)(lh, v);
        for (int k = 1; k <= MIN(u, n-j-1); k++)
            X(apply_reflector)(lh, v, tau[j], D+u-k+(j+k)*nu);
    }
    X(banded_qr) * F = malloc(sizeof(X(banded_qr)));
    F->factors = R;
    F->tau = tau;
    return F;
}

// x ← Q*x, x ← Qᵀ*x
void X(bqmv)(char TRANS, X(banded_qr) * F, FLT * x) {
    X(banded) * R = F->factors;
    FLT * D = R->data;
    FLT * tau = F->tau;
    int m = R->m, n = R->n, l = R->l, u = R->u;
    int nu = l+u+1;
    if (TRANS == 'N') {
        for (int j = MIN(m, n) - 1; j >= 0; j--) {
            int lh = MIN(l+1, m-j+1);
            X(apply_reflector)(lh, D+u+j*nu, tau[j], x+j);
        }
    }
    else if (TRANS == 'T') {
        for (int j = 0; j < MIN(m, n); j++) {
            int lh = MIN(l+1, m-j+1);
            X(apply_reflector)(lh, D+u+j*nu, tau[j], x+j);
        }
    }
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
            if (Y(fabs)(d) < 4*kd*Y(eps)() && Y(fabs)(t) < 4*kt*Y(eps)())
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
        #pragma omp parallel for num_threads(MIN(b, FT_GET_MAX_THREADS()))
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

        #pragma omp parallel for num_threads(MIN(b1+b2, FT_GET_MAX_THREADS()))
        for (int j = 0; j < b1+b2; j++)
            if (j < b1)
                X(bfmv)('T', F->F2, Y+j*(n-s));
            else
                X(bfmv)('T', F->F2, Y2+(j-b1)*(n-s));

       for (int j = 0; j < b2; j++)
             for (int i = 0; i < n-s; i++)
                 Y2[i+j*(n-s)] *= lambda[i+s];

        for (int j = 0; j < b2; j++)
            for (int i = 0; i < n-s; i++)
                Y[i+(j+b-b2)*(n-s)] = Y[i+(j+b-b2)*(n-s)]-Y2[i+j*(n-s)];

        int * p1 = malloc(s*sizeof(int));
        for (int i = 0; i < s; i++)
            p1[i] = i;
        X(quicksort_1arg)(lambda, p1, 0, s-1, X(lt));
        int * p2 = malloc((n-s)*sizeof(int));
        for (int i = 0; i < n-s; i++)
            p2[i] = i;
        X(quicksort_1arg)(lambda+s, p2, 0, n-s-1, X(lt));

        F->F0 = X(sample_hierarchicalmatrix)(X(cauchykernel), lambda, lambda, i, j, 'G');
        F->X = X;
        F->Y = Y;
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
        if (TRANS == 'N') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n-s; i++)
                    t2[i] = F->Y[p2[i]+k*(n-s)]*x[p2[i]+s];
                X(ghmv)(TRANS, -1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[p1[i]] += t1[i]*F->X[p1[i]+k*s];
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
                    t1[i] = F->X[p1[i]+k*s]*x[p1[i]];
                X(ghmv)(TRANS, -1, F->F0, t1, 0, t2);
                for (int i = 0; i < n-s; i++)
                    x[p2[i]+s] += t2[i]*F->Y[p2[i]+k*(n-s)];
            }
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
