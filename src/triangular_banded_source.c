void X(destroy_triangular_banded)(X(triangular_banded) * A) {
    free(A->data);
    free(A);
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
    int n = A->n;
    int b = A->b;
    if (0 <= j-i && j-i <= b && i < n && j < n)
        return data[b+i-j+j*(b+1)];
    else
        return ZERO(FLT);
}

void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j) {
    FLT * data = A->data;
    int n = A->n;
    int b = A->b;
    if (0 <= j-i && j-i <= b && i < n && j < n)
        data[b+i-j+j*(b+1)] = v;
}

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda) {
    for (int j = 0; j < A->n; j++)
        lambda[j] = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
}

// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V) {
    int n = A->n;
    int b1 = A->b;
    int b2 = B->b;
    int b = MAX(b1, b2);
    FLT t, lam;
    for (int j = 1; j < n; j++) {
        lam = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
        for (int i = j-1; i >= 0; i--) {
            t = ZERO(FLT);
            for (int k = i+1; k < MIN(i+b+1, n); k++)
                t += (X(get_triangular_banded_index)(A, i, k) - lam*X(get_triangular_banded_index)(B, i, k))*V[k+j*n];
            V[i+j*n] = t/(lam*X(get_triangular_banded_index)(B, i, i) - X(get_triangular_banded_index)(A, i, i));
        }
    }
}
