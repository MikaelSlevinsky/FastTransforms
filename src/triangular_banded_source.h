typedef struct {
    FLT * data;
    int n;
    int b;
} X(triangular_banded);

typedef struct X(tbstruct_FMM) X(tb_eigen_FMM);

struct X(tbstruct_FMM) {
    X(hierarchicalmatrix) * F0;
    X(tb_eigen_FMM) * F1;
    X(tb_eigen_FMM) * F2;
    FLT * V;
    FLT * X;
    FLT * Y;
    FLT * t1;
    FLT * t2;
    FLT * lambda;
    int n;
    int b;
};

void X(destroy_triangular_banded)(X(triangular_banded) * A);
void X(destroy_tb_eigen_FMM)(X(tb_eigen_FMM) * F);

size_t X(summary_size_tb_eigen_FMM)(X(tb_eigen_FMM) * F);

X(triangular_banded) * X(malloc_triangular_banded)(const int n, const int b);
X(triangular_banded) * X(calloc_triangular_banded)(const int n, const int b);

FLT X(get_triangular_banded_index)(const X(triangular_banded) * A, const int i, const int j);
void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j);

void X(tbmv)(char TRANS, X(triangular_banded) * A, FLT * x);
void X(tbsv)(char TRANS, X(triangular_banded) * A, FLT * x);

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda);
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V);

X(tb_eigen_FMM) * X(tb_eig_FMM)(X(triangular_banded) * A, X(triangular_banded) * B);

void X(scale_rows_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F);
void X(scale_columns_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F);

void X(trmv)(char TRANS, int n, FLT * A, FLT * x);
void X(trsv)(char TRANS, int n, FLT * A, FLT * x);

void X(bfmv)(char TRANS, X(tb_eigen_FMM) * A, FLT * x);
void X(bfsv)(char TRANS, X(tb_eigen_FMM) * A, FLT * x);

X(triangular_banded) * X(create_A_jacobi_to_jacobi)(const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta);
X(triangular_banded) * X(create_B_jacobi_to_jacobi)(const int n, const FLT gamma, const FLT delta);
