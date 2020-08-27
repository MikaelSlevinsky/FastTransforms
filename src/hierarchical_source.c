FLT * X(chebyshev_points)(char KIND, int n) {
    int nd2 = n>>1;
    FLT * x = malloc(n*sizeof(FLT));
    if (KIND == '1') {
        for (int k = 0; k <=nd2; k++)
            x[k] = Y(__sinpi)((n-2*k-ONE(FLT))/(2*n));
        for (int k = 0; k < nd2; k++)
            x[n-1-k] = -x[k];
    }
    else if (KIND == '2') {
        for (int k = 0; k <=nd2; k++)
            x[k] = Y(__sinpi)((n-2*k-ONE(FLT))/(2*n-2));
        for (int k = 0; k < nd2; k++)
            x[n-1-k] = -x[k];
    }
    return x;
}

FLT * X(chebyshev_barycentric_weights)(char KIND, int n) {
    int nd2 = n>>1;
    FLT * l = malloc(n*sizeof(FLT));
    if (KIND == '1') {
        for (int k = 0; k <=nd2; k++)
            l[k] = Y(__sinpi)((2*k+ONE(FLT))/(2*n));
        for (int k = 0; k < nd2; k++)
            l[n-1-k] = l[k];
        for (int k = 1; k < n; k += 2)
            l[k] *= -1;
    }
    else if (KIND == '2') {
        l[0] = ONE(FLT)/TWO(FLT);
        for (int k = 1; k <=nd2; k++)
            l[k] = 1;
        for (int k = 0; k < nd2; k++)
            l[n-1-k] = l[k];
        for (int k = 1; k < n; k += 2)
            l[k] *= -1;
    }
    return l;
}

// Evaluate a polynomial interpolant of degree n-1 through the distinct points
// y_j, 0 ≤ j < n at the distinct points x_i, 0 ≤ i < m. This is effected by a
// matrix-vector product with A_{i,j} and the polynomial interpolant's ordinates.
void * X(barycentricmatrix)(FLT * A, FLT * x, int m, FLT * y, FLT * l, int n) {
    int k;
    FLT yj, lj, temp;
    for (int j = 0; j < n; j++) {
        yj = y[j];
        lj = l[j];
        for (int i = 0; i < m; i++)
            A[i+m*j] = lj/(x[i]-yj);
    }
    for (int i = 0; i < m; i++) {
        k = -1;
        temp = 0;
        for (int j = 0; j < n; j++) {
            if (Y(isfinite)(A[i+m*j])) temp += A[i+m*j];
            else {k = j; break;}
        }
        if (k != -1) {
            for (int j = 0; j < n; j++)
                A[i+m*j] = 0;
            A[i+m*k] = 1;
        }
        else {
            temp = 1/temp;
            for (int j = 0; j < n; j++)
                A[i+m*j] *= temp;
        }
    }
}


void X(destroy_densematrix)(X(densematrix) * A) {
    free(A->A);
    free(A);
}

void X(destroy_lowrankmatrix)(X(lowrankmatrix) * L) {
    free(L->U);
    free(L->S);
    free(L->V);
    free(L->t1);
    free(L->t2);
    free(L);
}

void X(destroy_hierarchicalmatrix)(X(hierarchicalmatrix) * H) {
    int M = H->M, N = H->N;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            switch (H->hash(m, n)) {
                case 1: X(destroy_hierarchicalmatrix)(H->hierarchicalmatrices(m, n)); break;
                case 2: X(destroy_densematrix)(H->densematrices(m, n)); break;
                case 3: X(destroy_lowrankmatrix)(H->lowrankmatrices(m, n)); break;
            }
        }
    }
    free(H->hierarchicalmatrices);
    free(H->densematrices);
    free(H->lowrankmatrices);
    free(H->hash);
    free(H);
}


X(densematrix) * X(calloc_densematrix)(int m, int n) {
    X(densematrix) * A = malloc(sizeof(X(densematrix)));
    A->A = calloc(m*n, sizeof(FLT));
    A->m = m;
    A->n = n;
    return A;
}

X(densematrix) * X(malloc_densematrix)(int m, int n) {
    X(densematrix) * A = malloc(sizeof(X(densematrix)));
    A->A = malloc(m*n*sizeof(FLT));
    A->m = m;
    A->n = n;
    return A;
}

X(densematrix) * X(sample_densematrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j) {
    int M = i.stop-i.start;
    X(densematrix) * AD = X(malloc_densematrix)(M, j.stop-j.start);
    FLT * A = AD->A;
    for (int n = j.start; n < j.stop; n++)
        for (int m = i.start; m < i.stop; m++)
            A[m-i.start+M*(n-j.start)] = f(x[m], y[n]);
    return AD;
}

X(densematrix) * X(sample_accurately_densematrix)(FLT (*f)(FLT x, FLT ylo, FLT yhi), FLT * x, FLT * ylo, FLT * yhi, unitrange i, unitrange j) {
    int M = i.stop-i.start;
    X(densematrix) * AD = X(malloc_densematrix)(M, j.stop-j.start);
    FLT * A = AD->A;
    for (int n = j.start; n < j.stop; n++)
        for (int m = i.start; m < i.stop; m++)
            A[m-i.start+M*(n-j.start)] = f(x[m], ylo[n], yhi[n]);
    return AD;
}

void X(set_densematrix_index)(X(densematrix)* A, const FLT v, const int i, const int j){
    A -> A[i + j*A->m] = v;
}

FLT X(get_densematrix_index)(const X(densematrix)* A, const int i, const int j){
    return A -> A[i + j*A->m];
}

X(lowrankmatrix) * X(calloc_lowrankmatrix)(char N, int m, int n, int r) {
    int sz = 0;
    if (N == '2') sz = r;
    else if (N == '3') sz = r*r;
    X(lowrankmatrix) * L = malloc(sizeof(X(lowrankmatrix)));
    L->U = calloc(m*r, sizeof(FLT));
    L->S = calloc(sz, sizeof(FLT));
    L->V = calloc(n*r, sizeof(FLT));
    L->t1 = calloc(r*FT_GET_MAX_THREADS(), sizeof(FLT));
    L->t2 = calloc(r*FT_GET_MAX_THREADS(), sizeof(FLT));
    L->m = m;
    L->n = n;
    L->r = r;
    L->p = FT_GET_MAX_THREADS();
    L->N = N;
    return L;
}

X(lowrankmatrix) * X(malloc_lowrankmatrix)(char N, int m, int n, int r) {
    int sz = 0;
    if (N == '2') sz = r;
    else if (N == '3') sz = r*r;
    X(lowrankmatrix) * L = malloc(sizeof(X(lowrankmatrix)));
    L->U = malloc(m*r*sizeof(FLT));
    L->S = malloc(sz*sizeof(FLT));
    L->V = malloc(n*r*sizeof(FLT));
    L->t1 = calloc(r*FT_GET_MAX_THREADS(), sizeof(FLT));
    L->t2 = calloc(r*FT_GET_MAX_THREADS(), sizeof(FLT));
    L->m = m;
    L->n = n;
    L->r = r;
    L->p = FT_GET_MAX_THREADS();
    L->N = N;
    return L;
}

X(lowrankmatrix) * X(sample_lowrankmatrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j) {
    int M = i.stop-i.start, N = j.stop-j.start, r = BLOCKRANK;
    X(lowrankmatrix) * L = X(malloc_lowrankmatrix)('3', M, N, r);

    FLT * xc1 = X(chebyshev_points)('1', r);
    FLT * xc2 = X(chebyshev_points)('1', r);
    FLT * lc = X(chebyshev_barycentric_weights)('1', r);

    FLT a = x[i.start], b = x[i.stop-1];
    FLT c = y[j.start], d = y[j.stop-1];
    FLT ab2 = (a+b)/2, ba2 = (b-a)/2;
    FLT cd2 = (c+d)/2, dc2 = (d-c)/2;

    for (int p = 0; p < r; p++)
        xc1[p] = ab2+ba2*xc1[p];
    for (int q = 0; q < r; q++)
        xc2[q] = cd2+dc2*xc2[q];

    for (int q = 0; q < r; q++)
        for (int p = 0; p < r; p++)
            L->S[p+r*q] = f(xc1[p], xc2[q]);

    X(barycentricmatrix)(L->U, x+i.start, M, xc1, lc, r);
    X(barycentricmatrix)(L->V, y+j.start, N, xc2, lc, r);

    free(xc1);
    free(xc2);
    free(lc);

    return L;
}


X(hierarchicalmatrix) * X(malloc_hierarchicalmatrix)(const int M, const int N) {
    X(hierarchicalmatrix) * H = malloc(sizeof(X(hierarchicalmatrix)));
    H->hierarchicalmatrices = malloc(M*N*sizeof(X(hierarchicalmatrix) *));
    H->densematrices = malloc(M*N*sizeof(X(densematrix) *));
    H->lowrankmatrices = malloc(M*N*sizeof(X(lowrankmatrix) *));
    H->hash = calloc(M*N, sizeof(int));
    H->M = M;
    H->N = N;
    return H;
}

// Assumes x and y are increasing sequences
static FLT X(dist)(FLT * x, FLT * y, unitrange i, unitrange j) {
    if (y[j.start] > x[i.stop-1])
        return y[j.start] - x[i.stop-1];
    else if (y[j.start] >= x[i.start])
        return ZERO(FLT);
    else if (y[j.stop-1] >= x[i.start])
        return ZERO(FLT);
    else
        return x[i.start] - y[j.stop-1];
}

// Assumes x is an increasing sequence
static FLT X(diam)(FLT * x, unitrange i) {return x[i.stop-1] - x[i.start];}

X(hierarchicalmatrix) * X(sample_hierarchicalmatrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j, char SPLITTING) {
    int M = 2, N = 2;
    X(hierarchicalmatrix) * H = X(malloc_hierarchicalmatrix)(M, N);
    X(hierarchicalmatrix) ** HH = H->hierarchicalmatrices;
    X(densematrix) ** HD = H->densematrices;
    X(lowrankmatrix) ** HL = H->lowrankmatrices;

    unitrange i1, i2, j1, j2;
    if (SPLITTING == 'I') {
        i1.start = i.start;
        i1.stop = i2.start = i.start + (i.stop-i.start>>1);
        i2.stop = i.stop;
        j1.start = j.start;
        j1.stop = j2.start = j.start + (j.stop-j.start>>1);
        j2.stop = j.stop;
    }
    else if (SPLITTING == 'G') {
        X(indsplit)(x, i, &i1, &i2, x[i.start], x[i.stop-1]);
        X(indsplit)(y, j, &j1, &j2, y[j.start], y[j.stop-1]);
    }

    if (i1.stop-i1.start < BLOCKSIZE || j1.stop-j1.start < BLOCKSIZE) {
        HD[0] = X(sample_densematrix)(f, x, y, i1, j1);
        H->hash(0, 0) = 2;
    }
    else if (X(dist)(x, y, i1, j1) >= MIN(X(diam)(x, i1), X(diam)(y, j1))) {
        HL[0] = X(sample_lowrankmatrix)(f, x, y, i1, j1);
        H->hash(0, 0) = 3;
    }
    else {
        HH[0] = X(sample_hierarchicalmatrix)(f, x, y, i1, j1, SPLITTING);
        H->hash(0, 0) = 1;
    }

    if (i2.stop-i2.start < BLOCKSIZE || j1.stop-j1.start < BLOCKSIZE) {
        HD[1] = X(sample_densematrix)(f, x, y, i2, j1);
        H->hash(1, 0) = 2;
    }
    else if (X(dist)(x, y, i2, j1) >= MIN(X(diam)(x, i2), X(diam)(y, j1))) {
        HL[1] = X(sample_lowrankmatrix)(f, x, y, i2, j1);
        H->hash(1, 0) = 3;
    }
    else {
        HH[1] = X(sample_hierarchicalmatrix)(f, x, y, i2, j1, SPLITTING);
        H->hash(1, 0) = 1;
    }

    if (i1.stop-i1.start < BLOCKSIZE || j2.stop-j2.start < BLOCKSIZE) {
        HD[2] = X(sample_densematrix)(f, x, y, i1, j2);
        H->hash(0, 1) = 2;
    }
    else if (X(dist)(x, y, i1, j2) >= MIN(X(diam)(x, i1), X(diam)(y, j2))) {
        HL[2] = X(sample_lowrankmatrix)(f, x, y, i1, j2);
        H->hash(0, 1) = 3;
    }
    else {
        HH[2] = X(sample_hierarchicalmatrix)(f, x, y, i1, j2, SPLITTING);
        H->hash(0, 1) = 1;
    }

    if (i2.stop-i2.start < BLOCKSIZE || j2.stop-j2.start < BLOCKSIZE) {
        HD[3] = X(sample_densematrix)(f, x, y, i2, j2);
        H->hash(1, 1) = 2;
    }
    else if (X(dist)(x, y, i2, j2) >= MIN(X(diam)(x, i2), X(diam)(y, j2))) {
        HL[3] = X(sample_lowrankmatrix)(f, x, y, i2, j2);
        H->hash(1, 1) = 3;
    }
    else {
        HH[3] = X(sample_hierarchicalmatrix)(f, x, y, i2, j2, SPLITTING);
        H->hash(1, 1) = 1;
    }

    H->m = i.stop-i.start;
    H->n = j.stop-j.start;

    return H;
}

X(hierarchicalmatrix) * X(sample_accurately_hierarchicalmatrix)(FLT (*f)(FLT x, FLT y), FLT (*f2)(FLT x, FLT ylo, FLT yhi), FLT * x, FLT * y, FLT * ylo, FLT * yhi, unitrange i, unitrange j, char SPLITTING) {
    int M = 2, N = 2;
    X(hierarchicalmatrix) * H = X(malloc_hierarchicalmatrix)(M, N);
    X(hierarchicalmatrix) ** HH = H->hierarchicalmatrices;
    X(densematrix) ** HD = H->densematrices;
    X(lowrankmatrix) ** HL = H->lowrankmatrices;

    unitrange i1, i2, j1, j2;
    if (SPLITTING == 'I') {
        i1.start = i.start;
        i1.stop = i2.start = i.start + (i.stop-i.start>>1);
        i2.stop = i.stop;
        j1.start = j.start;
        j1.stop = j2.start = j.start + (j.stop-j.start>>1);
        j2.stop = j.stop;
    }
    else if (SPLITTING == 'G') {
        X(indsplit)(x, i, &i1, &i2, x[i.start], x[i.stop-1]);
        X(indsplit)(y, j, &j1, &j2, y[j.start], y[j.stop-1]);
    }

    if (i1.stop-i1.start < BLOCKSIZE || j1.stop-j1.start < BLOCKSIZE) {
        HD[0] = X(sample_accurately_densematrix)(f2, x, ylo, yhi, i1, j1);
        H->hash(0, 0) = 2;
    }
    else if (X(dist)(x, y, i1, j1) >= MIN(X(diam)(x, i1), X(diam)(y, j1))) {
        HL[0] = X(sample_lowrankmatrix)(f, x, y, i1, j1);
        H->hash(0, 0) = 3;
    }
    else {
        HH[0] = X(sample_accurately_hierarchicalmatrix)(f, f2, x, y, ylo, yhi, i1, j1, SPLITTING);
        H->hash(0, 0) = 1;
    }

    if (i2.stop-i2.start < BLOCKSIZE || j1.stop-j1.start < BLOCKSIZE) {
        HD[1] = X(sample_accurately_densematrix)(f2, x, ylo, yhi, i2, j1);
        H->hash(1, 0) = 2;
    }
    else if (X(dist)(x, y, i2, j1) >= MIN(X(diam)(x, i2), X(diam)(y, j1))) {
        HL[1] = X(sample_lowrankmatrix)(f, x, y, i2, j1);
        H->hash(1, 0) = 3;
    }
    else {
        HH[1] = X(sample_accurately_hierarchicalmatrix)(f, f2, x, y, ylo, yhi, i2, j1, SPLITTING);
        H->hash(1, 0) = 1;
    }

    if (i1.stop-i1.start < BLOCKSIZE || j2.stop-j2.start < BLOCKSIZE) {
        HD[2] = X(sample_accurately_densematrix)(f2, x, ylo, yhi, i1, j2);
        H->hash(0, 1) = 2;
    }
    else if (X(dist)(x, y, i1, j2) >= MIN(X(diam)(x, i1), X(diam)(y, j2))) {
        HL[2] = X(sample_lowrankmatrix)(f, x, y, i1, j2);
        H->hash(0, 1) = 3;
    }
    else {
        HH[2] = X(sample_accurately_hierarchicalmatrix)(f, f2, x, y, ylo, yhi, i1, j2, SPLITTING);
        H->hash(0, 1) = 1;
    }

    if (i2.stop-i2.start < BLOCKSIZE || j2.stop-j2.start < BLOCKSIZE) {
        HD[3] = X(sample_accurately_densematrix)(f2, x, ylo, yhi, i2, j2);
        H->hash(1, 1) = 2;
    }
    else if (X(dist)(x, y, i2, j2) >= MIN(X(diam)(x, i2), X(diam)(y, j2))) {
        HL[3] = X(sample_lowrankmatrix)(f, x, y, i2, j2);
        H->hash(1, 1) = 3;
    }
    else {
        HH[3] = X(sample_accurately_hierarchicalmatrix)(f, f2, x, y, ylo, yhi, i2, j2, SPLITTING);
        H->hash(1, 1) = 1;
    }

    H->m = i.stop-i.start;
    H->n = j.stop-j.start;

    return H;
}


static inline int X(size_densematrix)(X(densematrix) * A, int k) {
    if (k == 1) return A->m;
    else if (k == 2) return A->n;
    else return 1;
}

static inline int X(size_lowrankmatrix)(X(lowrankmatrix) * L, int k) {
    if (k == 1) return L->m;
    else if (k == 2) return L->n;
    else return 1;
}

static inline int X(size_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int k) {
    if (k == 1) return H->m;
    else if (k == 2) return H->n;
    else return 1;
}

static inline int X(blocksize_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int m, int n, int k) {
    int M = H->M, N = H->N;
    switch (H->hash(m, n)) {
        case 0: return 1;
        case 1: return X(size_hierarchicalmatrix)(H->hierarchicalmatrices(m, n), k);
        case 2: return X(size_densematrix)(H->densematrices(m, n), k);
        case 3: return X(size_lowrankmatrix)(H->lowrankmatrices(m, n), k);
    }
}

size_t X(summary_size_densematrix)(X(densematrix) * A) {return sizeof(FLT)*A->m*A->n;};
size_t X(summary_size_lowrankmatrix)(X(lowrankmatrix) * L) {return L->N == '2' ? sizeof(FLT)*(L->m+L->n+1)*L->r : L->N == '3' ? sizeof(FLT)*(L->m+L->n+L->r)*L->r : 0;};
size_t X(summary_size_hierarchicalmatrix)(X(hierarchicalmatrix) * H) {
    size_t M = H->M, N = H->N, S = 0;
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
            switch (H->hash(m, n)) {
                case 1: S += X(summary_size_hierarchicalmatrix)(H->hierarchicalmatrices(m, n)); break;
                case 2: S += X(summary_size_densematrix)(H->densematrices(m, n)); break;
                case 3: S += X(summary_size_lowrankmatrix)(H->lowrankmatrices(m, n)); break;
            }
    return S;
}

int X(nlevels_hierarchicalmatrix)(X(hierarchicalmatrix) * H) {
    int M = H->M, N = H->N, L = 0;
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
            if (H->hash(m, n) == 1)
                L = MAX(L, 1+X(nlevels_hierarchicalmatrix)(H->hierarchicalmatrices(m, n)));
    return L;
}

static inline FLT X(norm2_densematrix)(X(densematrix) * A) {
    int m = A->m, n = A->n;
    FLT ret = 0, * M = A->A;
    for (int i = 0; i < m*n; i++)
        ret += M[i]*M[i];
    return ret;
}

// ||UVᵀ||_F² = tr(UᵀU VᵀV) or ||USVᵀ||_F² = tr(UᵀUS VᵀVSᵀ)
static inline FLT X(norm2_lowrankmatrix)(X(lowrankmatrix) * L) {
    FLT ret = 0;
    int m = L->m, n = L->n, r = L->r;
    FLT * U = L->U, * S = L->S, * V = L->V;
    if (L->N == '2') {
        for (int l = 0; l < r; l++) {
            for (int k = 0; k < r; k++) {
                FLT tU = 0, tV = 0;
                for (int i = 0; i < m; i++)
                    tU += U[i+l*m]*U[i+k*m];
                for (int j = 0; j < n; j++)
                    tV += V[j+k*n]*V[j+l*n];
                ret += tU*tV;
            }
        }
    }
    else if (L->N == '3') {
        for (int l = 0; l < r; l++) {
            for (int k = 0; k < r; k++) {
                FLT tUS = 0, tVS = 0;
                for (int j = 0; j < r; j++) {
                    FLT tU = 0;
                    for (int i = 0; i < m; i++)
                        tU += U[i+l*m]*U[i+j*m];
                    tUS += tU*S[j+k*r];
                }
                for (int i = 0; i < r; i++) {
                    FLT tV = 0;
                    for (int j = 0; j < n; j++)
                        tV += V[j+k*n]*V[j+i*n];
                    tVS += tV*S[l+i*r];
                }
                ret += tUS*tVS;
            }
        }
    }
    return ret;
}

static inline FLT X(norm2_hierarchicalmatrix)(X(hierarchicalmatrix) * H) {
    int M = H->M, N = H->N;
    FLT ret = 0;
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
            switch (H->hash(m, n)) {
                case 1: ret += X(norm2_hierarchicalmatrix)(H->hierarchicalmatrices(m, n)); break;
                case 2: ret += X(norm2_densematrix)(H->densematrices(m, n)); break;
                case 3: ret += X(norm2_lowrankmatrix)(H->lowrankmatrices(m, n)); break;
            }
    return ret;
}

FLT X(norm_densematrix)(X(densematrix) * A) {return Y(sqrt)(X(norm2_densematrix)(A));}
FLT X(norm_lowrankmatrix)(X(lowrankmatrix) * L) {return Y(sqrt)(X(norm2_lowrankmatrix)(L));}
FLT X(norm_hierarchicalmatrix)(X(hierarchicalmatrix) * H) {return Y(sqrt)(X(norm2_hierarchicalmatrix)(H));}

void X(scale_rows_densematrix)(FLT alpha, FLT * x, X(densematrix) * AD) {
    int m = AD->m, n = AD->n;
    FLT * A = AD->A;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++)
            A[i] *= alpha*x[i];
        A += m;
    }
}

void X(scale_columns_densematrix)(FLT alpha, FLT * x, X(densematrix) * AD) {
    int m = AD->m, n = AD->n;
    FLT * A = AD->A;
    FLT axj;
    for (int j = 0; j < n; j++) {
        axj = alpha*x[j];
        for (int i = 0; i < m; i++)
            A[i] *= axj;
        A += m;
    }
}

void X(scale_rows_lowrankmatrix)(FLT alpha, FLT * x, X(lowrankmatrix) * L) {
    int m = L->m, r = L->r;
    FLT * U = L->U;
    for (int j = 0; j < r; j++) {
        for (int i = 0; i < m; i++)
            U[i] *= alpha*x[i];
        U += m;
    }
}

void X(scale_columns_lowrankmatrix)(FLT alpha, FLT * x, X(lowrankmatrix) * L) {
    int n = L->n, r = L->r;
    FLT * V = L->V;
    for (int j = 0; j < r; j++) {
        for (int i = 0; i < n; i++)
            V[i] *= alpha*x[i];
        V += n;
    }
}

void X(scale_rows_hierarchicalmatrix)(FLT alpha, FLT * x, X(hierarchicalmatrix) * H) {
    int M = H->M, N = H->N;
    for (int n = 0; n < N; n++) {
        int p = 0;
        for (int m = 0; m < M; m++) {
            switch (H->hash(m, n)) {
                case 1: X(scale_rows_hierarchicalmatrix)(alpha, x+p, H->hierarchicalmatrices(m, n)); break;
                case 2: X(scale_rows_densematrix)(alpha, x+p, H->densematrices(m, n)); break;
                case 3: X(scale_rows_lowrankmatrix)(alpha, x+p, H->lowrankmatrices(m, n)); break;
            }
            p += X(blocksize_hierarchicalmatrix)(H, m, N-1, 1);
        }
    }
}

void X(scale_columns_hierarchicalmatrix)(FLT alpha, FLT * x, X(hierarchicalmatrix) * H) {
    int M = H->M, N = H->N;
    int q = 0;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            switch (H->hash(m, n)) {
                case 1: X(scale_columns_hierarchicalmatrix)(alpha, x+q, H->hierarchicalmatrices(m, n)); break;
                case 2: X(scale_columns_densematrix)(alpha, x+q, H->densematrices(m, n)); break;
                case 3: X(scale_columns_lowrankmatrix)(alpha, x+q, H->lowrankmatrices(m, n)); break;
            }
        }
        q += X(blocksize_hierarchicalmatrix)(H, 0, n, 2);
    }
}


// y ← α*A*x + β*y, y ← α*Aᵀ*x + β*y
void X(gemv)(char TRANS, int m, int n, FLT alpha, FLT * A, int LDA, FLT * x, FLT beta, FLT * y) {
    FLT t;
    if (TRANS == 'N') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < m; i++)
                    y[i] = 0;
            else
                for (int i = 0; i < m; i++)
                    y[i] = beta*y[i];
        }
        for (int j = 0; j < n; j++) {
            t = alpha*x[j];
            for (int i = 0; i < m; i++)
                y[i] += A[i+j*LDA]*t;
        }
    }
    else if (TRANS == 'T') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < n; i++)
                    y[i] = 0;
            else
                for (int i = 0; i < n; i++)
                    y[i] = beta*y[i];
        }
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int j = 0; j < m; j++)
                t += A[j+i*LDA]*x[j];
            y[i] += alpha*t;
        }
    }
}

// C ← α*A*B + β*C, C ← α*Aᵀ*B + β*C
#if defined(FT_USE_CBLAS_S)
    void X(gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT * A, int LDA, FLT * B, int LDB, FLT beta, FLT * C, int LDC) {
        if (TRANS == 'N')
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, p, n, alpha, A, LDA, B, LDB, beta, C, LDC);
        else if (TRANS == 'T')
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, p, m, alpha, A, LDA, B, LDB, beta, C, LDC);
    }
#elif defined(FT_USE_CBLAS_D)
    void X(gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT * A, int LDA, FLT * B, int LDB, FLT beta, FLT * C, int LDC) {
        if (TRANS == 'N')
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, p, n, alpha, A, LDA, B, LDB, beta, C, LDC);
        else if (TRANS == 'T')
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, p, m, alpha, A, LDA, B, LDB, beta, C, LDC);
    }
#else
    void X(gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT * A, int LDA, FLT * B, int LDB, FLT beta, FLT * C, int LDC) {
        FLT t;
        if (TRANS == 'N') {
            for (int k = 0; k < p; k++) {
                if (beta != 1) {
                    if (beta == 0)
                        for (int i = 0; i < m; i++)
                            C[i+k*LDC] = 0;
                    else
                        for (int i = 0; i < m; i++)
                            C[i+k*LDC] = beta*C[i+k*LDC];
                }
                for (int j = 0; j < n; j++) {
                    t = alpha*B[j+k*LDB];
                    for (int i = 0; i < m; i++)
                        C[i+k*LDC] += A[i+j*LDA]*t;
                }
            }
        }
        else if (TRANS == 'T') {
            for (int k = 0; k < p; k++) {
                if (beta != 1) {
                    if (beta == 0)
                        for (int i = 0; i < n; i++)
                            C[i+k*LDC] = 0;
                    else
                        for (int i = 0; i < n; i++)
                            C[i+k*LDC] = beta*C[i+k*LDC];
                }
                for (int i = 0; i < n; i++) {
                    t = 0;
                    for (int j = 0; j < m; j++)
                        t += A[j+i*LDA]*B[j+k*LDB];
                    C[i+k*LDC] += alpha*t;
                }
            }
        }
    }
#endif


void X(demv)(char TRANS, FLT alpha, X(densematrix) * A, FLT * x, FLT beta, FLT * y) {
    X(gemv)(TRANS, A->m, A->n, alpha, A->A, A->m, x, beta, y);
}

void X(demm)(char TRANS, int p, FLT alpha, X(densematrix) * A, FLT * B, int LDB, FLT beta, FLT * C, int LDC) {
    X(gemm)(TRANS, A->m, A->n, p, alpha, A->A, A->m, B, LDB, beta, C, LDC);
}

// y ← α*(USVᵀ)*x + β*y, y ← α*(VSᵀUᵀ)*x + β*y
void X(lrmv)(char TRANS, FLT alpha, X(lowrankmatrix) * L, FLT * x, FLT beta, FLT * y) {
    int m = L->m, n = L->n, r = L->r;
    FLT * t1 = L->t1+r*FT_GET_THREAD_NUM(), * t2 = L->t2+r*FT_GET_THREAD_NUM();
    if (TRANS == 'N') {
        if (L->N == '2') {
            X(gemv)('T', n, r, 1, L->V, n, x, 0, t1);
            X(gemv)('N', m, r, alpha, L->U, m, t1, beta, y);
        }
        else if (L->N == '3') {
            X(gemv)('T', n, r, 1, L->V, n, x, 0, t1);
            X(gemv)('N', r, r, 1, L->S, r, t1, 0, t2);
            X(gemv)('N', m, r, alpha, L->U, m, t2, beta, y);
        }
    }
    else if (TRANS == 'T') {
        if (L->N == '2') {
            X(gemv)('T', m, r, 1, L->U, m, x, 0, t1);
            X(gemv)('N', n, r, alpha, L->V, n, t1, beta, y);
        }
        else if (L->N == '3') {
            X(gemv)('T', m, r, 1, L->U, m, x, 0, t1);
            X(gemv)('T', r, r, 1, L->S, r, t1, 0, t2);
            X(gemv)('N', n, r, alpha, L->V, n, t2, beta, y);
        }
    }
}

static inline void X(check_temps_lowrankmatrix)(X(lowrankmatrix) * L, int p) {
    if (L->p < p) {
        L->t1 = realloc(L->t1, L->r*p*sizeof(FLT));
        L->t2 = realloc(L->t2, L->r*p*sizeof(FLT));
        L->p = p;
    }
}

// C ← α*(USVᵀ)*B + β*C, C ← α*(VSᵀUᵀ)*B + β*C
void X(lrmm)(char TRANS, int p, FLT alpha, X(lowrankmatrix) * L, FLT * B, int LDB, FLT beta, FLT * C, int LDC) {
    X(check_temps_lowrankmatrix)(L, p);
    int m = L->m, n = L->n, r = L->r;
    FLT * t1 = L->t1, * t2 = L->t2;
    if (TRANS == 'N') {
        if (L->N == '2') {
            X(gemm)('T', n, r, p, 1, L->V, n, B, LDB, 0, t1, r);
            X(gemm)('N', m, r, p, alpha, L->U, m, t1, r, beta, C, LDC);
        }
        else if (L->N == '3') {
            X(gemm)('T', n, r, p, 1, L->V, n, B, LDB, 0, t1, r);
            X(gemm)('N', r, r, p, 1, L->S, r, t1, r, 0, t2, r);
            X(gemm)('N', m, r, p, alpha, L->U, m, t2, r, beta, C, LDC);
        }
    }
    else if (TRANS == 'T') {
        if (L->N == '2') {
            X(gemm)('T', m, r, p, 1, L->U, m, B, LDB, 0, t1, r);
            X(gemm)('N', n, r, p, alpha, L->V, n, t1, r, beta, C, LDC);
        }
        else if (L->N == '3') {
            X(gemm)('T', m, r, p, 1, L->U, m, B, LDB, 0, t1, r);
            X(gemm)('T', r, r, p, 1, L->S, r, t1, r, 0, t2, r);
            X(gemm)('N', n, r, p, alpha, L->V, n, t2, r, beta, C, LDC);
        }
    }
}

// y ← α*H*x + β*y, y ← α*Hᵀ*x + β*y
void X(ghmv)(char TRANS, FLT alpha, X(hierarchicalmatrix) * H, FLT * x, FLT beta, FLT * y) {
    int M = H->M, N = H->N;
    int p, q = 0;
    if (TRANS == 'N') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < X(size_hierarchicalmatrix)(H, 1); i++)
                    y[i] = 0;
            else
                for (int i = 0; i < X(size_hierarchicalmatrix)(H, 1); i++)
                    y[i] = beta*y[i];
        }
        for (int n = 0; n < N; n++) {
            p = 0;
            for (int m = 0; m < M; m++) {
                switch (H->hash(m, n)) {
                    case 1: X(ghmv)(TRANS, alpha, H->hierarchicalmatrices(m, n), x+q, 1, y+p); break;
                    case 2: X(demv)(TRANS, alpha, H->densematrices(m, n),        x+q, 1, y+p); break;
                    case 3: X(lrmv)(TRANS, alpha, H->lowrankmatrices(m, n),      x+q, 1, y+p); break;
                }
                p += X(blocksize_hierarchicalmatrix)(H, m, N-1, 1);
            }
            q += X(blocksize_hierarchicalmatrix)(H, 0, n, 2);
        }
    }
    else if (TRANS == 'T') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < X(size_hierarchicalmatrix)(H, 2); i++)
                    y[i] = 0;
            else
                for (int i = 0; i < X(size_hierarchicalmatrix)(H, 2); i++)
                    y[i] = beta*y[i];
        }
        for (int m = 0; m < M; m++) {
            p = 0;
            for (int n = 0; n < N; n++) {
                switch (H->hash(m, n)) {
                    case 1: X(ghmv)(TRANS, alpha, H->hierarchicalmatrices(m, n), x+q, 1, y+p); break;
                    case 2: X(demv)(TRANS, alpha, H->densematrices(m, n),        x+q, 1, y+p); break;
                    case 3: X(lrmv)(TRANS, alpha, H->lowrankmatrices(m, n),      x+q, 1, y+p); break;
                }
                p += X(blocksize_hierarchicalmatrix)(H, 0, n, 2);
            }
            q += X(blocksize_hierarchicalmatrix)(H, m, N-1, 1);
        }
    }
}

// C ← α*H*B + β*C, C ← α*Hᵀ*B + β*C
void X(ghmm)(char TRANS, int p, FLT alpha, X(hierarchicalmatrix) * H, FLT * B, int LDB, FLT beta, FLT * C, int LDC) {
    int M = H->M, N = H->N, P = 2;
    int pcols[] = {0, p/P, p};
    int * mrows = calloc(M, sizeof(int));
    for (int m = 1; m < M; m++)
        mrows[m] = mrows[m-1] + X(blocksize_hierarchicalmatrix)(H, m-1, N-1, 1);
    int * ncols = calloc(N, sizeof(int));
    for (int n = 1; n < N; n++)
        ncols[n] = ncols[n-1] + X(blocksize_hierarchicalmatrix)(H, 0, n-1, 2);
    if (TRANS == 'N') {
        if (beta != 1) {
            if (beta == 0) {
                #pragma omp parallel for
                for (int j = 0; j < p; j++)
                    for (int i = 0; i < X(size_hierarchicalmatrix)(H, 1); i++)
                        C[i+j*LDC] = 0;
            }
            else {
                #pragma omp parallel for
                for (int j = 0; j < p; j++)
                    for (int i = 0; i < X(size_hierarchicalmatrix)(H, 1); i++)
                        C[i+j*LDC] = beta*C[i+j*LDC];
            }
        }
        if (p >= X(size_hierarchicalmatrix)(H, 2)) {
            for (int p = 0; p < P; p++) {
                #pragma omp parallel for
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        switch (H->hash(m, n)) {
                            case 1: X(ghmm)(TRANS, pcols[p+1]-pcols[p], alpha, H->hierarchicalmatrices(m, n), B+ncols[n]+pcols[p]*LDB, LDB, 1, C+mrows[m]+pcols[p]*LDC, LDC); break;
                            case 2: X(demm)(TRANS, pcols[p+1]-pcols[p], alpha, H->densematrices(m, n),        B+ncols[n]+pcols[p]*LDB, LDB, 1, C+mrows[m]+pcols[p]*LDC, LDC); break;
                            case 3: X(lrmm)(TRANS, pcols[p+1]-pcols[p], alpha, H->lowrankmatrices(m, n),      B+ncols[n]+pcols[p]*LDB, LDB, 1, C+mrows[m]+pcols[p]*LDC, LDC); break;
                        }
                    }
                }
            }
        }
        else {
            #pragma omp parallel for
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    switch (H->hash(m, n)) {
                        case 1: X(ghmm)(TRANS, p, alpha, H->hierarchicalmatrices(m, n), B+ncols[n], LDB, 1, C+mrows[m], LDC); break;
                        case 2: X(demm)(TRANS, p, alpha, H->densematrices(m, n),        B+ncols[n], LDB, 1, C+mrows[m], LDC); break;
                        case 3: X(lrmm)(TRANS, p, alpha, H->lowrankmatrices(m, n),      B+ncols[n], LDB, 1, C+mrows[m], LDC); break;
                    }
                }
            }
        }
    }
    else if (TRANS == 'T') {
        if (beta != 1) {
            if (beta == 0) {
                #pragma omp parallel for
                for (int j = 0; j < p; j++)
                    for (int i = 0; i < X(size_hierarchicalmatrix)(H, 2); i++)
                        C[i+j*LDC] = 0;
            }
            else {
                #pragma omp parallel for
                for (int j = 0; j < p; j++)
                    for (int i = 0; i < X(size_hierarchicalmatrix)(H, 2); i++)
                        C[i+j*LDC] = beta*C[i+j*LDC];
            }
        }
        if (p >= X(size_hierarchicalmatrix)(H, 1)) {
            for (int p = 0; p < P; p++) {
                #pragma omp parallel for
                for (int n = 0; n < N; n++) {
                    for (int m = 0; m < M; m++) {
                        switch (H->hash(m, n)) {
                            case 1: X(ghmm)(TRANS, pcols[p+1]-pcols[p], alpha, H->hierarchicalmatrices(m, n), B+mrows[m]+pcols[p]*LDB, LDB, 1, C+ncols[n]+pcols[p]*LDC, LDC); break;
                            case 2: X(demm)(TRANS, pcols[p+1]-pcols[p], alpha, H->densematrices(m, n),        B+mrows[m]+pcols[p]*LDB, LDB, 1, C+ncols[n]+pcols[p]*LDC, LDC); break;
                            case 3: X(lrmm)(TRANS, pcols[p+1]-pcols[p], alpha, H->lowrankmatrices(m, n),      B+mrows[m]+pcols[p]*LDB, LDB, 1, C+ncols[n]+pcols[p]*LDC, LDC); break;
                        }
                    }
                }
            }
        }
        else {
            #pragma omp parallel for
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    switch (H->hash(m, n)) {
                        case 1: X(ghmm)(TRANS, p, alpha, H->hierarchicalmatrices(m, n), B+mrows[m], LDB, 1, C+ncols[n], LDC); break;
                        case 2: X(demm)(TRANS, p, alpha, H->densematrices(m, n),        B+mrows[m], LDB, 1, C+ncols[n], LDC); break;
                        case 3: X(lrmm)(TRANS, p, alpha, H->lowrankmatrices(m, n),      B+mrows[m], LDB, 1, C+ncols[n], LDC); break;
                    }
                }
            }
        }
    }
    free(mrows);
    free(ncols);
}


int X(binarysearch)(FLT * x, int start, int stop, FLT y) {
    int j;
    while (stop >= start) {
        j = (start+stop)/2;
        if (x[j] < y) start = j+1;
        else if (x[j] > y) stop = j-1;
        else break;
    }
    if (x[j] < y) j += 1;
    return j;
}

/*
indsplit takes a unitrange `start ≤ ir < stop`, and splits it into
two unitranges `i1` and `i2` such that

    `a ≤ x[i] < (a+b)/2` for `i ∈ i1`, and
    `(a+b)/2 ≤ x[i] ≤ b` for `i ∈ i2`.

*/
void X(indsplit)(FLT * x, unitrange ir, unitrange * i1, unitrange * i2, FLT a, FLT b) {
    int start = ir.start, stop = ir.stop;
    i1->start = start;
    i1->stop = i2->start = X(binarysearch)(x, start, stop, (a+b)/2);
    i2->stop = stop;
}

FLT X(cauchykernel)(FLT x, FLT y) {return 1/(x-y);}
FLT X(coulombkernel)(FLT x, FLT y) {return 1/((x-y)*(x-y));}
FLT X(coulombprimekernel)(FLT x, FLT y) {return 1/(((x-y)*(x-y))*(x-y));}
FLT X(logkernel)(FLT x, FLT y) {return Y(log)(Y(fabs)(x-y));}

static FLT X(diff)(FLT x, FLT y) {return x - y;}

FLT X(cauchykernel2)(FLT x, FLT ylo, FLT yhi) {return 1/(X(diff)(x, yhi) - ylo);}
FLT X(coulombkernel2)(FLT x, FLT ylo, FLT yhi) {return 1/((X(diff)(x, yhi) - ylo)*(X(diff)(x, yhi) - ylo));}
FLT X(coulombprimekernel2)(FLT x, FLT ylo, FLT yhi) {return 1/(((X(diff)(x, yhi) - ylo)*(X(diff)(x, yhi) - ylo))*(X(diff)(x, yhi) - ylo));}
FLT X(logkernel2)(FLT x, FLT ylo, FLT yhi) {return Y(log)(Y(fabs)(X(diff)(x, yhi) - ylo));}
