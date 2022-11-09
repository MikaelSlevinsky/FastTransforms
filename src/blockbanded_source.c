void X(destroy_block_2x2_triangular_banded)(X(block_2x2_triangular_banded) * A) {
    X(destroy_triangular_banded)(A->data[0][0]);
    X(destroy_triangular_banded)(A->data[0][1]);
    X(destroy_triangular_banded)(A->data[1][0]);
    X(destroy_triangular_banded)(A->data[1][1]);
    free(A);
}

void X(destroy_btb_eigen_FMM)(X(btb_eigen_FMM) * F) {
    X(destroy_tb_eigen_FMM)(F->F);
    free(F->s);
    free(F->c);
    free(F->t);
    free(F);
}

X(block_2x2_triangular_banded) * X(create_block_2x2_triangular_banded)(X(triangular_banded) * data[2][2]) {
    X(block_2x2_triangular_banded) * A = malloc(sizeof(X(block_2x2_triangular_banded)));
    int n = data[0][0]->n;
    if (data[0][1]->n != n || data[1][0]->n != n || data[1][1]->n != n)
        exit_failure("create_block_2x2_triangular_banded: block sizes are not all the same.");
    int b = MAX(MAX(data[0][0]->b, data[0][1]->b), MAX(data[1][0]->b, data[1][1]->b));
    if (data[0][0]->b != b)
        X(realloc_triangular_banded)(data[0][0], b);
    if (data[0][1]->b != b)
        X(realloc_triangular_banded)(data[0][1], b);
    if (data[1][0]->b != b)
        X(realloc_triangular_banded)(data[1][0], b);
    if (data[1][1]->b != b)
        X(realloc_triangular_banded)(data[1][1], b);
    A->data[0][0] = data[0][0];
    A->data[0][1] = data[0][1];
    A->data[1][0] = data[1][0];
    A->data[1][1] = data[1][1];
    A->n = n;
    A->b = b;
    return A;
}

X(triangular_banded) * X(convert_block_2x2_triangular_banded_to_triangular_banded)(X(block_2x2_triangular_banded) * A) {
    int n = A->n, b = A->b;
    X(triangular_banded) * B = X(malloc_triangular_banded)(2*n, 2*b+1);
    for (int j = 0; j < 2*n; j++)
        for (int k = MAX(j-2*b-1, 0); k <= j; k++)
            X(set_triangular_banded_index)(B, X(get_block_2x2_triangular_banded_index)(A, k, j), k, j);
    return B;
}

FLT X(get_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, const int i, const int j) {
    return X(get_triangular_banded_index)(A->data[i%2][j%2], i/2, j/2);
}

void X(set_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, const FLT v, const int i, const int j) {
    return X(set_triangular_banded_index)(A->data[i%2][j%2], v, i/2, j/2);
}

void X(block_get_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, FLT v[2][2], const int i, const int j) {
    v[0][0] = X(get_triangular_banded_index)(A->data[0][0], i, j);
    v[0][1] = X(get_triangular_banded_index)(A->data[0][1], i, j);
    v[1][0] = X(get_triangular_banded_index)(A->data[1][0], i, j);
    v[1][1] = X(get_triangular_banded_index)(A->data[1][1], i, j);
}

void X(block_set_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, const FLT v[2][2], const int i, const int j) {
    X(set_triangular_banded_index)(A->data[0][0], v[0][0], i, j);
    X(set_triangular_banded_index)(A->data[0][1], v[0][1], i, j);
    X(set_triangular_banded_index)(A->data[1][0], v[1][0], i, j);
    X(set_triangular_banded_index)(A->data[1][1], v[1][1], i, j);
}


static inline void X(inverse_2x2)(const FLT A[2][2], FLT B[2][2]) {
    FLT d = A[0][0]*A[1][1] - A[0][1]*A[1][0];
    B[0][0] = A[1][1]/d;
    B[0][1] = -A[0][1]/d;
    B[1][0] = -A[1][0]/d;
    B[1][1] = A[0][0]/d;
}

// x ← A*x, x ← Aᵀ*x
void X(btbmv)(char TRANS, X(block_2x2_triangular_banded) * A, FLT * x) {
    int n = A->n, bnd = A->b;
    FLT a[2][2], t[2];
    if (TRANS == 'N') {
        for (int i = 0; i < n; i++) {
            t[1] = t[0] = 0;
            for (int k = i; k < MIN(i+bnd+1, n); k++) {
                X(block_get_block_2x2_triangular_banded_index)(A, a, i, k);
                t[0] += a[0][0]*x[2*k] + a[0][1]*x[2*k+1];
                t[1] += a[1][0]*x[2*k] + a[1][1]*x[2*k+1];
            }
            x[2*i] = t[0];
            x[2*i+1] = t[1];
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            t[1] = t[0] = 0;
            for (int k = MAX(i-bnd, 0); k <= i; k++) {
                X(block_get_block_2x2_triangular_banded_index)(A, a, k, i);
                t[0] += a[0][0]*x[2*k] + a[1][0]*x[2*k+1];
                t[1] += a[0][1]*x[2*k] + a[1][1]*x[2*k+1];
            }
            x[2*i] = t[0];
            x[2*i+1] = t[1];
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(btbsv)(char TRANS, X(block_2x2_triangular_banded) * A, FLT * x) {
    int n = A->n, bnd = A->b;
    FLT a[2][2], b[2][2], t[2];
    if (TRANS == 'N') {
        for (int i = n-1; i >= 0; i--) {
            t[1] = t[0] = 0;
            for (int k = i+1; k < MIN(i+bnd+1, n); k++) {
                X(block_get_block_2x2_triangular_banded_index)(A, a, i, k);
                t[0] += a[0][0]*x[2*k] + a[0][1]*x[2*k+1];
                t[1] += a[1][0]*x[2*k] + a[1][1]*x[2*k+1];
            }
            X(block_get_block_2x2_triangular_banded_index)(A, a, i, i);
            X(inverse_2x2)(a, b);
            t[0] = x[2*i]-t[0];
            t[1] = x[2*i+1]-t[1];
            x[2*i] = b[0][0]*t[0] + b[0][1]*t[1];
            x[2*i+1] = b[1][0]*t[0] + b[1][1]*t[1];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            t[1] = t[0] = 0;
            for (int k = MAX(i-bnd, 0); k < i; k++) {
                X(block_get_block_2x2_triangular_banded_index)(A, a, k, i);
                t[0] += a[0][0]*x[2*k] + a[1][0]*x[2*k+1];
                t[1] += a[0][1]*x[2*k] + a[1][1]*x[2*k+1];
            }
            X(block_get_block_2x2_triangular_banded_index)(A, a, i, i);
            X(inverse_2x2)(a, b);
            t[0] = x[2*i]-t[0];
            t[1] = x[2*i+1]-t[1];
            x[2*i] = b[0][0]*t[0] + b[1][0]*t[1];
            x[2*i+1] = b[0][1]*t[0] + b[1][1]*t[1];
        }
    }
}

// AV = BVΛ, A and B are block upper-triangular and banded and Λ is real.

static inline void X(real_quadratic_formula)(const FLT a, const FLT b, const FLT c, FLT x[2]) {
    FLT d = b*b-4*a*c;
    if (d < 0)
        exit_failure("real_quadratic_formula: discriminant is negative.");
    d = Y(sqrt)(d);
    if (a > 0) {
        if (b > 0) {
            x[0] = -(b+d)/(2*a);
            x[1] = -2*c/(b+d);
        }
        else {
            x[0] = 2*c/(d-b);
            x[1] = (d-b)/(2*a);
        }
    }
    else if (a < 0) {
        if (b > 0) {
            x[0] = -2*c/(b+d);
            x[1] = -(b+d)/(2*a);
        }
        else {
            x[0] = (d-b)/(2*a);
            x[1] = 2*c/(d-b);
        }
    }
    else
        exit_failure("real_quadratic_formula: quadratic is a degenerate linear.");
}

static inline void X(generalized_eigenvalues_2x2)(const FLT A[2][2], const FLT B[2][2], FLT lambda[2]) {
    FLT a, b, c;
    a = B[0][0]*B[1][1]-B[0][1]*B[1][0];
    b = A[0][1]*B[1][0]+A[1][0]*B[0][1]-A[0][0]*B[1][1]-A[1][1]*B[0][0];
    c = A[0][0]*A[1][1]-A[0][1]*A[1][0];
    X(real_quadratic_formula)(a, b, c, lambda);
}

void X(block_2x2_triangular_banded_eigenvalues)(X(block_2x2_triangular_banded) * A, X(block_2x2_triangular_banded) * B, FLT * lambda) {
    FLT a[2][2], b[2][2];
    for (int j = 0; j < A->n; j++) {
        X(block_get_block_2x2_triangular_banded_index)(A, a, j, j);
        X(block_get_block_2x2_triangular_banded_index)(B, b, j, j);
        X(generalized_eigenvalues_2x2)(a, b, lambda+2*j);
    }
}

static inline void X(scaled_diff_2x2)(const FLT A[2][2], const FLT lambda, const FLT B[2][2], FLT C[2][2]) {
    C[0][0] = A[0][0] - lambda*B[0][0];
    C[0][1] = A[0][1] - lambda*B[0][1];
    C[1][0] = A[1][0] - lambda*B[1][0];
    C[1][1] = A[1][1] - lambda*B[1][1];
}

// Assumes eigenvectors are initialized by V[i,2j] = V[i,2j+1] = 0 for i > 2j+1 and V[2j,2j] ≠ 0, V[2j,2j+1] ≠ 0.
void X(block_2x2_triangular_banded_eigenvectors)(X(block_2x2_triangular_banded) * A, X(block_2x2_triangular_banded) * B, FLT * V) {
    int n = A->n, bnd = MAX(A->b, B->b);
    FLT t[2], a[2][2], b[2][2], c[2][2], d[2][2], lam[2];
    for (int j = 0; j < n; j++) {
        X(block_get_block_2x2_triangular_banded_index)(A, a, j, j);
        X(block_get_block_2x2_triangular_banded_index)(B, b, j, j);
        X(generalized_eigenvalues_2x2)(a, b, lam);
        V[2*j+1+2*j*2*n] = (b[1][0]*lam[0]-a[1][0])*V[2*j+2*j*2*n]/(a[1][1]-b[1][1]*lam[0]);
        V[2*j+1+(2*j+1)*2*n] = (b[1][0]*lam[1]-a[1][0])*V[2*j+(2*j+1)*2*n]/(a[1][1]-b[1][1]*lam[1]);
        for (int i = j-1; i >= 0; i--) {
            for (int l = 0; l <= 1; l++) {
                t[1] = t[0] = 0;
                for (int k = i+1; k < MIN(i+bnd+1, n); k++) {
                    X(block_get_block_2x2_triangular_banded_index)(A, a, i, k);
                    X(block_get_block_2x2_triangular_banded_index)(B, b, i, k);
                    X(scaled_diff_2x2)(a, lam[l], b, c);
                    t[0] += c[0][0]*V[2*k+(2*j+l)*2*n] + c[0][1]*V[2*k+1+(2*j+l)*2*n];
                    t[1] += c[1][0]*V[2*k+(2*j+l)*2*n] + c[1][1]*V[2*k+1+(2*j+l)*2*n];
                }
                X(block_get_block_2x2_triangular_banded_index)(A, a, i, i);
                X(block_get_block_2x2_triangular_banded_index)(B, b, i, i);
                X(scaled_diff_2x2)(a, lam[l], b, c);
                X(inverse_2x2)(c, d);
                V[2*i+(2*j+l)*2*n] = -(d[0][0]*t[0] + d[0][1]*t[1]);
                V[2*i+1+(2*j+l)*2*n] = -(d[1][0]*t[0] + d[1][1]*t[1]);
            }
        }
    }
}

// Apply Givens rotation [c s; -s c] or its TRANSpose to array A from the left or right SIDE.
static inline void X(apply_givens)(char TRANS, char SIDE, const FLT c, const FLT s, FLT A[2][2]) {
    FLT t1, t2, t3, t4;
    if (SIDE == 'L') {
        if (TRANS == 'N') {
            t1 = c*A[0][0]+s*A[1][0];
            t2 = c*A[0][1]+s*A[1][1];
            t3 = c*A[1][0]-s*A[0][0];
            t4 = c*A[1][1]-s*A[0][1];
            A[0][0] = t1;
            A[0][1] = t2;
            A[1][0] = t3;
            A[1][1] = t4;
        }
        else if (TRANS == 'T') {
            t1 = c*A[0][0]-s*A[1][0];
            t2 = c*A[0][1]-s*A[1][1];
            t3 = c*A[1][0]+s*A[0][0];
            t4 = c*A[1][1]+s*A[0][1];
            A[0][0] = t1;
            A[0][1] = t2;
            A[1][0] = t3;
            A[1][1] = t4;
        }
    }
    else if (SIDE == 'R') {
        if (TRANS == 'N') {
            t1 = c*A[0][0]-s*A[0][1];
            t2 = c*A[0][1]+s*A[0][0];
            t3 = c*A[1][0]-s*A[1][1];
            t4 = c*A[1][1]+s*A[1][0];
            A[0][0] = t1;
            A[0][1] = t2;
            A[1][0] = t3;
            A[1][1] = t4;
        }
        else if (TRANS == 'T') {
            t1 = c*A[0][0]+s*A[0][1];
            t2 = c*A[0][1]-s*A[0][0];
            t3 = c*A[1][0]+s*A[1][1];
            t4 = c*A[1][1]-s*A[1][0];
            A[0][0] = t1;
            A[0][1] = t2;
            A[1][0] = t3;
            A[1][1] = t4;
        }
    }
}

// D is 2n initial conditions.
// On entry: D[2j] = V[2j, 2j], D[2j+1] = V[2j, 2j+1].
// On exit: D[2j] = TV[2j, 2j], D[2j+1] = TV[2j+1, 2j+1].
X(btb_eigen_FMM) * X(btb_eig_FMM)(X(block_2x2_triangular_banded) * A, X(block_2x2_triangular_banded) * B, FLT * D) {
    int n = A->n, bnd = MAX(A->b, B->b);
    FLT * s = malloc(n*sizeof(FLT));
    FLT * c = malloc(n*sizeof(FLT));
    FLT a[2][2], b[2][2], lambda[2], ts, tc, r, t1, t2;
    // Stage 1: triangularize (2x2 block-triangular) eigenvectors via Givens rotations.
    for (int j = 0; j < n; j++) {
        X(block_get_block_2x2_triangular_banded_index)(A, a, j, j);
        X(block_get_block_2x2_triangular_banded_index)(B, b, j, j);
        X(generalized_eigenvalues_2x2)(a, b, lambda);
        t1 = (b[1][0]*lambda[0]-a[1][0])*D[2*j]/(a[1][1]-b[1][1]*lambda[0]);
        t2 = (b[1][0]*lambda[1]-a[1][0])*D[2*j+1]/(a[1][1]-b[1][1]*lambda[1]);
        X(compute_givens)(D[2*j], t1, c+j, s+j, &r);
        D[2*j] = r;
        D[2*j+1] = c[j]*t2-s[j]*D[2*j+1];
        for (int k = MAX(j-bnd, 0); k <= j; k++) {
            X(block_get_block_2x2_triangular_banded_index)(A, a, k, j);
            X(apply_givens)('T', 'R', c[j], s[j], a);
            X(block_set_block_2x2_triangular_banded_index)(A, a, k, j);
            X(block_get_block_2x2_triangular_banded_index)(B, b, k, j);
            X(apply_givens)('T', 'R', c[j], s[j], b);
            X(block_set_block_2x2_triangular_banded_index)(B, b, k, j);
        }
    }
    // Stage 2: triangularize (2x2 block-triangular banded) pencil via Givens rotations.
    for (int i = 0; i < n; i++) {
        X(block_get_block_2x2_triangular_banded_index)(B, b, i, i);
        X(compute_givens)(b[0][0], b[1][0], &tc, &ts, &r);
        for (int k = i; k < MIN(i+bnd+1, n); k++) {
            X(block_get_block_2x2_triangular_banded_index)(A, a, i, k);
            X(apply_givens)('N', 'L', tc, ts, a);
            X(block_set_block_2x2_triangular_banded_index)(A, a, i, k);
            X(block_get_block_2x2_triangular_banded_index)(B, b, i, k);
            X(apply_givens)('N', 'L', tc, ts, b);
            X(block_set_block_2x2_triangular_banded_index)(B, b, i, k);
        }
    }
    // Stage 3: convert (2x2 block-triangular banded) pencil to triangular banded pencil.
    X(triangular_banded) * TA = X(convert_block_2x2_triangular_banded_to_triangular_banded)(A);
    X(triangular_banded) * TB = X(convert_block_2x2_triangular_banded_to_triangular_banded)(B);
    // Stage 4: call X(tb_eig_FMM)(TA, TB, TD)
    X(tb_eigen_FMM) * F = X(tb_eig_FMM)(TA, TB, D);
    X(destroy_triangular_banded)(TA);
    X(destroy_triangular_banded)(TB);
    X(btb_eigen_FMM) * BF = malloc(sizeof(X(btb_eigen_FMM)));
    BF->F = F;
    BF->s = s;
    BF->c = c;
    BF->t = calloc(2*n*FT_GET_MAX_THREADS(), sizeof(FLT));
    BF->n = n;
    return BF;
}


// x ← A*x, x ← Aᵀ*x
void X(btrmv)(char TRANS, int n, FLT * A, int LDA, FLT * x) {
    FLT t[2];
    if (TRANS == 'N') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++) {
                x[2*i] += A[2*i+2*j*LDA]*x[2*j] + A[2*i+(2*j+1)*LDA]*x[2*j+1];
                x[2*i+1] += A[2*i+1+2*j*LDA]*x[2*j] + A[2*i+1+(2*j+1)*LDA]*x[2*j+1];
            }
            t[0] = x[2*j];
            t[1] = x[2*j+1];
            x[2*j] = A[2*j+2*j*LDA]*t[0] + A[2*j+(2*j+1)*LDA]*t[1];
            x[2*j+1] = A[2*j+1+2*j*LDA]*t[0] + A[2*j+1+(2*j+1)*LDA]*t[1];
        }
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i >= 0; i--) {
            t[0] = x[2*i];
            t[1] = x[2*i+1];
            x[2*i] = A[2*i+2*i*LDA]*t[0] + A[2*i+1+2*i*LDA]*t[1];
            x[2*i+1] = A[2*i+(2*i+1)*LDA]*t[0] + A[2*i+1+(2*i+1)*LDA]*t[1];
            for (int j = i-1; j >= 0; j--) {
                x[2*i] += A[2*j+2*i*LDA]*x[2*j] + A[2*j+1+2*i*LDA]*x[2*j+1];
                x[2*i+1] += A[2*j+(2*i+1)*LDA]*x[2*j] + A[2*j+1+(2*i+1)*LDA]*x[2*j+1];
            }
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(btrsv)(char TRANS, int n, FLT * A, int LDA, FLT * x) {
    FLT c[2][2], d[2][2], t[2];
    if (TRANS == 'N') {
        for (int j = n-1; j >= 0; j--) {
            c[0][0] = A[2*j+2*j*LDA];
            c[0][1] = A[2*j+(2*j+1)*LDA];
            c[1][0] = A[2*j+1+2*j*LDA];
            c[1][1] = A[2*j+1+(2*j+1)*LDA];
            X(inverse_2x2)(c, d);
            t[0] = x[2*j];
            t[1] = x[2*j+1];
            x[2*j] = d[0][0]*t[0] + d[0][1]*t[1];
            x[2*j+1] = d[1][0]*t[0] + d[1][1]*t[1];
            for (int i = 0; i < j; i++) {
                x[2*i] -= A[2*i+2*j*LDA]*x[2*j] + A[2*i+(2*j+1)*LDA]*x[2*j+1];
                x[2*i+1] -= A[2*i+1+2*j*LDA]*x[2*j] + A[2*i+1+(2*j+1)*LDA]*x[2*j+1];
            }
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                x[2*i] -= A[2*j+2*i*LDA]*x[2*j] + A[2*j+1+2*i*LDA]*x[2*j+1];
                x[2*i+1] -= A[2*j+(2*i+1)*LDA]*x[2*j] + A[2*j+1+(2*i+1)*LDA]*x[2*j+1];
            }
            c[0][0] = A[2*i+2*i*LDA];
            c[0][1] = A[2*i+1+2*i*LDA];
            c[1][0] = A[2*i+(2*i+1)*LDA];
            c[1][1] = A[2*i+1+(2*i+1)*LDA];
            X(inverse_2x2)(c, d);
            t[0] = x[2*i];
            t[1] = x[2*i+1];
            x[2*i] = d[0][0]*t[0] + d[0][1]*t[1];
            x[2*i+1] = d[1][0]*t[0] + d[1][1]*t[1];
        }
    }
}

// B ← A*B, B ← Aᵀ*B
void X(btrmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(btrmv)(TRANS, n, A, LDA, B+j*LDB);
}

// B ← A⁻¹*B, B ← A⁻ᵀ*B
void X(btrsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(btrsv)(TRANS, n, A, LDA, B+j*LDB);
}

// x ← A*x, x ← Aᵀ*x
void X(bbfmv)(char TRANS, X(btb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    FLT * s = F->s, * c = F->c, t1, t2;
    if (TRANS == 'N') {
        // Apply upper-triangular part
        X(bfmv)(TRANS, F->F, x);
        // Apply Givens rotations
        for (int i = 0; i < n; i++) {
            t1 = c[i]*x[2*i]-s[i]*x[2*i+1];
            t2 = c[i]*x[2*i+1]+s[i]*x[2*i];
            x[2*i] = t1;
            x[2*i+1] = t2;
        }
    }
    else if (TRANS == 'T') {
        // Apply Givens rotations
        for (int i = 0; i < n; i++) {
            t1 = c[i]*x[2*i]+s[i]*x[2*i+1];
            t2 = c[i]*x[2*i+1]-s[i]*x[2*i];
            x[2*i] = t1;
            x[2*i+1] = t2;
        }
        // Apply upper-triangular part
        X(bfmv)(TRANS, F->F, x);
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(bbfsv)(char TRANS, X(btb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    FLT * s = F->s, * c = F->c, t1, t2;
    if (TRANS == 'N') {
        // Apply Givens rotations
        for (int i = 0; i < n; i++) {
            t1 = c[i]*x[2*i]+s[i]*x[2*i+1];
            t2 = c[i]*x[2*i+1]-s[i]*x[2*i];
            x[2*i] = t1;
            x[2*i+1] = t2;
        }
        // Apply upper-triangular part
        X(bfsv)(TRANS, F->F, x);
    }
    else if (TRANS == 'T') {
        // Apply upper-triangular part
        X(bfsv)(TRANS, F->F, x);
        // Apply Givens rotations
        for (int i = 0; i < n; i++) {
            t1 = c[i]*x[2*i]-s[i]*x[2*i+1];
            t2 = c[i]*x[2*i+1]+s[i]*x[2*i];
            x[2*i] = t1;
            x[2*i+1] = t2;
        }
    }
}

void X(bbfmm)(char TRANS, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bbfmv)(TRANS, F, B+j*LDB);
}

void X(bbfsm)(char TRANS, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bbfsv)(TRANS, F, B+j*LDB);
}

// x ← A*x, x ← Aᵀ*x
void X(bbbfmv)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    FLT * t = F->t+2*n*FT_GET_THREAD_NUM();
    if (DBLOCK == '1') {
        for (int i = 0; i < n; i++) {
            t[2*i] = x[i];
            t[2*i+1] = 0;
        }
    }
    else if (DBLOCK == '2') {
        for (int i = 0; i < n; i++) {
            t[2*i] = 0;
            t[2*i+1] = x[i];
        }
    }
    X(bbfmv)(TRANS, F, t);
    if (RBLOCK == '1')
        for (int i = 0; i < n; i++)
            x[i] = t[2*i];
    else if (RBLOCK == '2')
        for (int i = 0; i < n; i++)
            x[i] = t[2*i+1];
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(bbbfsv)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * x) {
    int n = F->n;
    FLT * t = F->t+2*n*FT_GET_THREAD_NUM();
    if (RBLOCK == '1') {
        for (int i = 0; i < n; i++) {
            t[2*i] = x[i];
            t[2*i+1] = 0;
        }
    }
    else if (RBLOCK == '2') {
        for (int i = 0; i < n; i++) {
            t[2*i] = 0;
            t[2*i+1] = x[i];
        }
    }
    X(bbfsv)(TRANS, F, t);
    if (DBLOCK == '1')
        for (int i = 0; i < n; i++)
            x[i] = t[2*i];
    else if (DBLOCK == '2')
        for (int i = 0; i < n; i++)
            x[i] = t[2*i+1];
}

void X(bbbfmm)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bbbfmv)(TRANS, DBLOCK, RBLOCK, F, B+j*LDB);
}

void X(bbbfsm)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N) {
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
        X(bbbfsv)(TRANS, DBLOCK, RBLOCK, F, B+j*LDB);
}
