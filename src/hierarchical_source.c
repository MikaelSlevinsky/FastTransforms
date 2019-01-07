FLT * X(chebyshev_points)(char KIND, int n) {
    int nd2 = n/2;
    FLT * x = (FLT *) malloc(n*sizeof(FLT));
    if (KIND == '1') {
        for (int k = 0; k < nd2+1; k++)
            x[k] = X(__sinpi)((n-2*k-ONE(FLT))/(2*n));
        for (int k = 0; k < nd2; k++)
            x[n-1-k] = -x[k];
    }
    else if (KIND == '2') {
        for (int k = 0; k < nd2+1; k++)
            x[k] = X(__sinpi)((n-2*k-ONE(FLT))/(2*n-2));
        for (int k = 0; k < nd2; k++)
            x[n-1-k] = -x[k];
    }
    return x;
}

FLT * X(chebyshev_barycentric_weights)(char KIND, int n) {
    int nd2 = n/2;
    FLT * l = (FLT *) malloc(n*sizeof(FLT));
    if (KIND == '1') {
        for (int k = 0; k < nd2+1; k++)
            l[k] = X(__sinpi)((2*k+ONE(FLT))/(2*n));
        for (int k = 0; k < nd2; k++)
            l[n-1-k] = l[k];
        for (int k = 1; k < n; k += 2)
            l[k] *= -1;
    }
    else if (KIND == '2') {
        l[0] = ONE(FLT)/TWO(FLT);
        for (int k = 1; k < nd2+1; k++)
            l[k] = ONE(FLT);
        for (int k = 0; k < nd2; k++)
            l[n-1-k] = l[k];
        for (int k = 1; k < n; k += 2)
            l[k] *= -1;
    }
    return l;
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
                case 1: {
                    X(destroy_hierarchicalmatrix)(H->hierarchicalmatrices(m, n));
                    break;
                }
                case 2: {
                    X(destroy_densematrix)(H->densematrices(m, n));
                    break;
                }
                case 3: {
                    X(destroy_lowrankmatrix)(H->lowrankmatrices(m, n));
                    break;
                }
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
    X(densematrix) * A = (X(densematrix) *) malloc(sizeof(X(densematrix)));
    A->A = (FLT *) calloc(m*n, sizeof(FLT));
    A->m = m;
    A->n = n;
    return A;
}

X(densematrix) * X(malloc_densematrix)(int m, int n) {
    X(densematrix) * A = (X(densematrix) *) malloc(sizeof(X(densematrix)));
    A->A = (FLT *) malloc(m*n*sizeof(FLT));
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

X(lowrankmatrix) * X(calloc_lowrankmatrix)(char N, int m, int n, int r) {
    int sz = 0;
    if (N == '2') sz = r;
    else if (N == '3') sz = r*r;
    X(lowrankmatrix) * L = (X(lowrankmatrix) *) malloc(sizeof(X(lowrankmatrix)));
    L->U = (FLT *) calloc(m*r, sizeof(FLT));
    L->S = (FLT *) calloc(sz, sizeof(FLT));
    L->V = (FLT *) calloc(n*r, sizeof(FLT));
    L->t1 = (FLT *) calloc(r, sizeof(FLT));
    L->t2 = (FLT *) calloc(r, sizeof(FLT));
    L->m = m;
    L->n = n;
    L->r = r;
    L->N = N;
    return L;
}

X(lowrankmatrix) * X(malloc_lowrankmatrix)(char N, int m, int n, int r) {
    int sz = 0;
    if (N == '2') sz = r;
    else if (N == '3') sz = r*r;
    X(lowrankmatrix) * L = (X(lowrankmatrix) *) malloc(sizeof(X(lowrankmatrix)));
    L->U = (FLT *) malloc(m*r*sizeof(FLT));
    L->S = (FLT *) malloc(sz*sizeof(FLT));
    L->V = (FLT *) malloc(n*r*sizeof(FLT));
    L->t1 = (FLT *) calloc(r, sizeof(FLT));
    L->t2 = (FLT *) calloc(r, sizeof(FLT));
    L->m = m;
    L->n = n;
    L->r = r;
    L->N = N;
    return L;
}

X(lowrankmatrix) * X(sample_lowrankmatrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j) {
    int M = i.stop-i.start, N = j.stop-j.start, r = BLOCKRANK, k;
    X(lowrankmatrix) * L = X(malloc_lowrankmatrix)('3', M, N, r);

    FLT xcpq, lcpq, temp;
    FLT * xc = X(chebyshev_points)('1', r);
    FLT * lc = X(chebyshev_barycentric_weights)('1', r);

    FLT a = x[i.start], b = x[i.stop-1];
    FLT c = y[j.start], d = y[j.stop-1];
    FLT ab2 = (a+b)/2, ba2 = (b-a)/2;
    FLT cd2 = (c+d)/2, dc2 = (d-c)/2;

    FLT * S = L->S, * U = L->U, * V = L->V;
    for (int q = 0; q < r; q++)
        for (int p = 0; p < r; p++)
            S[p+r*q] = f(ab2+ba2*xc[p], cd2+dc2*xc[q]);

    for (int p = 0; p < r; p++) {
        xcpq = ab2+ba2*xc[p];
        lcpq = lc[p];
        for (int m = i.start; m < i.stop; m++)
            U[m-i.start+M*p] = lcpq/(x[m]-xcpq);
    }
    for (int m = 0; m < i.stop-i.start; m++) {
        k = -1;
        temp = 0;
        for (int p = 0; p < r; p++) {
            if (X(isnan)(U[m+M*p]) || X(isinf)(U[m+M*p])) {k = p; break;}
            else temp += U[m+M*p];
        }
        if (k != -1) {
            for (int p = 0; p < r; p++)
                U[m+M*p] = 0;
            U[m+M*k] = 1;
        }
        else {
            temp = 1/temp;
            for (int p = 0; p < r; p++)
                U[m+M*p] *= temp;
        }
    }

    for (int q = 0; q < r; q++) {
        xcpq = cd2+dc2*xc[q];
        lcpq = lc[q];
        for (int n = j.start; n < j.stop; n++)
            V[n-j.start+N*q] = lcpq/(y[n]-xcpq);
    }
    for (int n = 0; n < j.stop-j.start; n++) {
        k = -1;
        temp = 0;
        for (int q = 0; q < r; q++) {
            if (X(isnan)(V[n+N*q]) || X(isinf)(V[n+N*q])) {k = q; break;}
            else temp += V[n+N*q];
        }
        if (k != -1) {
            for (int q = 0; q < r; q++)
                V[n+N*q] = 0;
            V[n+N*k] = 1;
        }
        else {
            temp = 1/temp;
            for (int q = 0; q < r; q++)
                V[n+N*q] *= temp;
        }
    }

    free(xc);
    free(lc);

    return L;
}


X(hierarchicalmatrix) * X(malloc_hierarchicalmatrix)(const int M, const int N) {
    X(hierarchicalmatrix) * H = (X(hierarchicalmatrix) *) malloc(sizeof(X(hierarchicalmatrix)));
    H->hierarchicalmatrices = (X(hierarchicalmatrix) **) malloc(M*N*sizeof(X(hierarchicalmatrix)));
    H->densematrices = (X(densematrix) **) malloc(M*N*sizeof(X(densematrix)));
    H->lowrankmatrices = (X(lowrankmatrix) **) malloc(M*N*sizeof(X(lowrankmatrix)));
    H->hash = (int *) calloc(M*N, sizeof(int));
    H->M = M;
    H->N = N;
    return H;
}

X(hierarchicalmatrix) * X(create_hierarchicalmatrix)(const int M, const int N, FLT (*f)(FLT x, FLT y), const int m, const int n) {
    X(hierarchicalmatrix) * H = X(malloc_hierarchicalmatrix)(M, N);
    X(densematrix) ** HD = H->densematrices;
    FLT * A;
    for (int nn = 0; nn < N; nn++) {
        for (int mm = 0; mm < M; mm++) {
            H->hash(mm,nn) = 2;
            HD[mm+M*nn] = X(calloc_densematrix)(m, n);
            A = HD[mm+M*nn]->A;
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    A[i+m*j] = f(i + mm*m, j+nn*n);
                }
            }
        }
    }
    return H;
}


X(hierarchicalmatrix) * X(sample_hierarchicalmatrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j) {
    int M = 2, N = 2;
    X(hierarchicalmatrix) * H = X(malloc_hierarchicalmatrix)(M, N);
    X(hierarchicalmatrix) ** HH = H->hierarchicalmatrices;
    X(densematrix) ** HD = H->densematrices;

    unitrange i1, i2, j1, j2;
    X(indsplit)(x, i, &i1, &i2, x[i.start], x[i.stop-1]);
    X(indsplit)(y, j, &j1, &j2, y[j.start], y[j.stop-1]);

    if ((i1.stop-i1.start)*(j1.stop-j1.start) < BLOCKSIZE*BLOCKSIZE) {
        HD[0] = X(sample_densematrix)(f, x, y, i1, j1);
        H->hash(0, 0) = 2;
    }
    else {
        HH[0] = X(sample_hierarchicalmatrix)(f, x, y, i1, j1);
        H->hash(0, 0) = 1;
    }

    if ((i2.stop-i2.start)*(j1.stop-j1.start) < BLOCKSIZE*BLOCKSIZE) {
        HD[1] = X(sample_densematrix)(f, x, y, i2, j1);
        H->hash(1, 0) = 2;
    }
    else {
        HH[1] = X(sample_hierarchicalmatrix1)(f, x, y, i2, j1);
        H->hash(1, 0) = 1;
    }

    if ((i1.stop-i1.start)*(j2.stop-j2.start) < BLOCKSIZE*BLOCKSIZE) {
        HD[2] = X(sample_densematrix)(f, x, y, i1, j2);
        H->hash(0, 1) = 2;
    }
    else {
        HH[2] = X(sample_hierarchicalmatrix2)(f, x, y, i1, j2);
        H->hash(0, 1) = 1;
    }

    if ((i2.stop-i2.start)*(j2.stop-j2.start) < BLOCKSIZE*BLOCKSIZE) {
        HD[3] = X(sample_densematrix)(f, x, y, i2, j2);
        H->hash(1, 1) = 2;
    }
    else {
        HH[3] = X(sample_hierarchicalmatrix)(f, x, y, i2, j2);
        H->hash(1, 1) = 1;
    }

    return H;
}

X(hierarchicalmatrix) * X(sample_hierarchicalmatrix1)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j) {
    int M = 2, N = 2;
    X(hierarchicalmatrix) * H = X(malloc_hierarchicalmatrix)(M, N);
    X(hierarchicalmatrix) ** HH = H->hierarchicalmatrices;
    X(densematrix) ** HD = H->densematrices;
    X(lowrankmatrix) ** HL = H->lowrankmatrices;

    unitrange i1, i2, j1, j2;
    X(indsplit)(x, i, &i1, &i2, x[i.start], x[i.stop-1]);
    X(indsplit)(y, j, &j1, &j2, y[j.start], y[j.stop-1]);

    HL[0] = X(sample_lowrankmatrix)(f, x, y, i1, j1);
    H->hash(0, 0) = 3;
    HL[1] = X(sample_lowrankmatrix)(f, x, y, i2, j1);
    H->hash(1, 0) = 3;
    if ((i1.stop-i1.start)*(j2.stop-j2.start) < BLOCKSIZE*BLOCKSIZE) {
        HD[2] = X(sample_densematrix)(f, x, y, i1, j2);
        H->hash(0, 1) = 2;
    }
    else {
        HH[2] = X(sample_hierarchicalmatrix1)(f, x, y, i1, j2);
        H->hash(0, 1) = 1;
    }
    HL[3] = X(sample_lowrankmatrix)(f, x, y, i2, j2);
    H->hash(1, 1) = 3;

    return H;
}

X(hierarchicalmatrix) * X(sample_hierarchicalmatrix2)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j) {
    int M = 2, N = 2;
    X(hierarchicalmatrix) * H = X(malloc_hierarchicalmatrix)(M, N);
    X(hierarchicalmatrix) ** HH = H->hierarchicalmatrices;
    X(densematrix) ** HD = H->densematrices;
    X(lowrankmatrix) ** HL = H->lowrankmatrices;

    unitrange i1, i2, j1, j2;
    X(indsplit)(x, i, &i1, &i2, x[i.start], x[i.stop-1]);
    X(indsplit)(y, j, &j1, &j2, y[j.start], y[j.stop-1]);

    HL[0] = X(sample_lowrankmatrix)(f, x, y, i1, j1);
    H->hash(0, 0) = 3;
    if ((i2.stop-i2.start)*(j1.stop-j1.start) < BLOCKSIZE*BLOCKSIZE) {
        HD[1] = X(sample_densematrix)(f, x, y, i2, j1);
        H->hash(1, 0) = 2;
    }
    else {
        HH[1] = X(sample_hierarchicalmatrix2)(f, x, y, i2, j1);
        H->hash(1, 0) = 1;
    }
    HL[2] = X(sample_lowrankmatrix)(f, x, y, i1, j2);
    H->hash(0, 1) = 3;
    HL[3] = X(sample_lowrankmatrix)(f, x, y, i2, j2);
    H->hash(1, 1) = 3;

    return H;
}


int X(size_densematrix)(X(densematrix) * A, int k) {
    if (k == 1) return A->m;
    else if (k == 2) return A->n;
    else return 1;
}

int X(size_lowrankmatrix)(X(lowrankmatrix) * L, int k) {
    if (k == 1) return L->m;
    else if (k == 2) return L->n;
    else return 1;
}

int X(size_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int k) {
    int M = H->M, N = H->N;
    if (k == 1) {
        int p = 0;
        for (int m = 0; m < M; m++)
            p += X(blocksize_hierarchicalmatrix)(H, m, N-1, 1);
        return p;
    }
    else if (k == 2) {
        int q = 0;
        for (int n = 0; n < N; n++)
            q += X(blocksize_hierarchicalmatrix)(H, 0, n, 2);
        return q;
    }
    else return 1;
}

int X(blocksize_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int m, int n, int k) {
    int M = H->M, N = H->N;
    switch (H->hash(m, n)) {
        case 0: return 1;
        case 1: return X(size_hierarchicalmatrix)(H->hierarchicalmatrices(m, n), k);
        case 2: return X(size_densematrix)(H->densematrices(m, n), k);
        case 3: return X(size_lowrankmatrix)(H->lowrankmatrices(m, n), k);
    }
}


// y ← α*A*x + β*y, y ← α*Aᵀ*x + β*y
void X(gemv)(char TRANS, int m, int n, FLT alpha, FLT * A, FLT * x, FLT beta, FLT * y) {
    FLT xj, t;
    if (TRANS == 'N') {
        for (int i = 0; i < m; i++)
            y[i] = beta*y[i];
        for (int j = 0; j < n; j++) {
            xj = alpha*x[j];
            for (int i = 0; i < m; i++)
                y[i] += A[i]*xj;
            A += m;
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++)
            y[i] = beta*y[i];
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int j = 0; j < m; j++)
                t += A[j]*x[j];
            y[i] += alpha*t;
            A += m;
        }
    }
}

void X(demv)(char TRANS, FLT alpha, X(densematrix) * A, FLT * x, FLT beta, FLT * y) {
    X(gemv)(TRANS, A->m, A->n, alpha, A->A, x, beta, y);
}

// y ← α*(USVᵀ)*x + β*y, y ← α*(VSᵀUᵀ)*x + β*y
void X(lrmv)(char TRANS, FLT alpha, X(lowrankmatrix) * L, FLT * x, FLT beta, FLT * y) {
    int m = L->m, n = L->n, r = L->r;
    FLT * t1 = L->t1, * t2 = L->t2;
    if (TRANS == 'N') {
        if (L->N == '2') {
            X(gemv)('T', n, r, ONE(FLT), L->V, x, ZERO(FLT), t1);
            X(gemv)('N', m, r, alpha, L->U, t1, beta, y);
        }
        else if (L->N == '3') {
            X(gemv)('T', n, r, ONE(FLT), L->V, x, ZERO(FLT), t1);
            X(gemv)('N', r, r, ONE(FLT), L->S, t1, ZERO(FLT), t2);
            X(gemv)('N', m, r, alpha, L->U, t2, beta, y);
        }
    }
    else if (TRANS == 'T') {
        if (L->N == '2') {
            X(gemv)('T', m, r, ONE(FLT), L->U, x, ZERO(FLT), t1);
            X(gemv)('N', n, r, alpha, L->V, t1, beta, y);
        }
        else if (L->N == '3') {
            X(gemv)('T', m, r, ONE(FLT), L->U, x, ZERO(FLT), t1);
            X(gemv)('T', r, r, ONE(FLT), L->S, t1, ZERO(FLT), t2);
            X(gemv)('N', n, r, alpha, L->V, t2, beta, y);
        }
    }
}

// y ← α*H*x + β*y, y ← α*Hᵀ*x + β*y
void X(himv)(char TRANS, FLT alpha, X(hierarchicalmatrix) * H, FLT * x, FLT beta, FLT * y) {
    int M = H->M, N = H->N;
    int p, q = 0;
    if (TRANS == 'N') {
        for (int i = 0; i < X(size_hierarchicalmatrix)(H, 1); i++)
            y[i] = beta*y[i];
        for (int n = 0; n < N; n++) {
            p = 0;
            for (int m = 0; m < M; m++) {
                switch (H->hash(m, n)) {
                    case 1: {
                        X(himv)(TRANS, alpha, H->hierarchicalmatrices(m, n), x+q, ONE(FLT), y+p);
                        break;
                    }
                    case 2: {
                        X(demv)(TRANS, alpha, H->densematrices(m, n), x+q, ONE(FLT), y+p);
                        break;
                    }
                    case 3: {
                        X(lrmv)(TRANS, alpha, H->lowrankmatrices(m, n), x+q, ONE(FLT), y+p);
                        break;
                    }
                }
                p += X(blocksize_hierarchicalmatrix)(H, m, N-1, 1);
            }
            q += X(blocksize_hierarchicalmatrix)(H, 0, n, 2);
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < X(size_hierarchicalmatrix)(H, 2); i++)
            y[i] = beta*y[i];
        for (int m = 0; m < M; m++) {
            p = 0;
            for (int n = 0; n < N; n++) {
                switch (H->hash(m, n)) {
                    case 1: {
                        X(himv)(TRANS, alpha, H->hierarchicalmatrices(m, n), x+q, ONE(FLT), y+p);
                        break;
                    }
                    case 2: {
                        X(demv)(TRANS, alpha, H->densematrices(m, n), x+q, ONE(FLT), y+p);
                        break;
                    }
                    case 3: {
                        X(lrmv)(TRANS, alpha, H->lowrankmatrices(m, n), x+q, ONE(FLT), y+p);
                        break;
                    }
                }
                p += X(blocksize_hierarchicalmatrix)(H, 0, n, 2);
            }
            q += X(blocksize_hierarchicalmatrix)(H, m, N-1, 1);
        }
    }
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
FLT X(logkernel)(FLT x, FLT y) {return X(log)(X(fabs)(x-y));}
