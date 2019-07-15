typedef struct {
    FLT * data;
    int n;
    int b;
} X(triangular_banded);

void X(destroy_triangular_banded)(X(triangular_banded) * A);

X(triangular_banded) * X(malloc_triangular_banded)(const int n, const int b);
X(triangular_banded) * X(calloc_triangular_banded)(const int n, const int b);

FLT X(get_triangular_banded_index)(const X(triangular_banded) * A, const int i, const int j);
void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j);

void X(tbmv)(char TRANS, X(triangular_banded) * A, FLT * x);
void X(tbsv)(char TRANS, X(triangular_banded) * A, FLT * x);

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda);
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V);

X(triangular_banded) * X(create_A_jac2jac)(const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta);
X(triangular_banded) * X(create_B_jac2jac)(const int n, const FLT gamma, const FLT delta);
