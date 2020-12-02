typedef struct {
    int * p;
    int * q;
    FLT * v;
    int m;
    int n;
    int nnz;
} X(sparse);

typedef struct {
    FLT * data;
    int m;
    int n;
    int l;
    int u;
} X(banded);

typedef struct {
    FLT * data;
    int n;
    int b;
} X(triangular_banded);

typedef struct {
    X(banded) * factors;
    FLT * tau;
} X(banded_qr);

typedef struct X(tbstruct_FMM) X(tb_eigen_FMM);

struct X(tbstruct_FMM) {
    X(hierarchicalmatrix) * F0;
    X(tb_eigen_FMM) * F1;
    X(tb_eigen_FMM) * F2;
    X(sparse) * S;
    FLT * V;
    FLT * X;
    FLT * Y;
    FLT * t1;
    FLT * t2;
    FLT * lambda;
    int * p1;
    int * p2;
    int n;
    int b;
};

typedef struct X(tbstruct_ADI) X(tb_eigen_ADI);

struct X(tbstruct_ADI) {
    X(lowrankmatrix) * F0;
    X(tb_eigen_ADI) * F1;
    X(tb_eigen_ADI) * F2;
    FLT * V;
    FLT * lambda;
    int n;
    int b;
};

void X(destroy_sparse)(X(sparse) * A);
void X(destroy_banded)(X(banded) * A);
void X(destroy_triangular_banded)(X(triangular_banded) * A);
void X(destroy_banded_qr)(X(banded_qr) * F);
void X(destroy_tb_eigen_FMM)(X(tb_eigen_FMM) * F);
void X(destroy_tb_eigen_ADI)(X(tb_eigen_ADI) * F);

size_t X(summary_size_tb_eigen_FMM)(X(tb_eigen_FMM) * F);
size_t X(summary_size_tb_eigen_ADI)(X(tb_eigen_ADI) * F);

X(sparse) * X(malloc_sparse)(const int m, const int n, const int nnz);
X(sparse) * X(calloc_sparse)(const int m, const int n, const int nnz);

X(banded) * X(malloc_banded)(const int m, const int n, const int l, const int u);
X(banded) * X(calloc_banded)(const int m, const int n, const int l, const int u);

X(triangular_banded) * X(malloc_triangular_banded)(const int n, const int b);
X(triangular_banded) * X(calloc_triangular_banded)(const int n, const int b);
void X(realloc_triangular_banded)(X(triangular_banded) * A, const int b);

X(triangular_banded) * X(view_triangular_banded)(const X(triangular_banded) * A, const unitrange i);

X(banded) * X(convert_triangular_banded_to_banded)(X(triangular_banded) * A);
X(triangular_banded) * X(convert_banded_to_triangular_banded)(X(banded) * A);

X(triangular_banded) * X(create_I_triangular_banded)(const int n, const int b);

FLT X(get_banded_index)(const X(banded) * A, const int i, const int j);
void X(set_banded_index)(const X(banded) * A, const FLT v, const int i, const int j);

FLT X(get_triangular_banded_index)(const X(triangular_banded) * A, const int i, const int j);
void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j);

void X(gbmv)(FLT alpha, X(banded) * A, FLT * x, FLT beta, FLT * y);
void X(gbmm)(FLT alpha, X(banded) * A, X(banded) * B, FLT beta, X(banded) * C);
void X(banded_add)(FLT alpha, X(banded) * A, FLT beta, X(banded) * B, X(banded) * C);

void X(tbmv)(char TRANS, X(triangular_banded) * A, FLT * x);
void X(tbsv)(char TRANS, X(triangular_banded) * A, FLT * x);
void X(tssv)(char TRANS, X(triangular_banded) * A, X(triangular_banded) * B, FLT gamma, FLT * x);

void X(banded_lufact)(X(banded) * A);
X(banded_qr) * X(banded_qrfact)(X(banded) * A);
void X(bqmv)(char TRANS, X(banded_qr) * F, FLT * x);
void X(brmv)(char TRANS, X(banded_qr) * F, FLT * x);
void X(brsv)(char TRANS, X(banded_qr) * F, FLT * x);

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda);
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V);

void X(triangular_banded_quadratic_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, X(triangular_banded) * C, FLT * lambda);
void X(triangular_banded_quadratic_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, X(triangular_banded) * C, FLT * V);

X(tb_eigen_FMM) * X(tb_eig_FMM)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * D);
X(tb_eigen_ADI) * X(tb_eig_ADI)(X(triangular_banded) * A, X(triangular_banded) * B);

void X(scale_rows_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F);
void X(scale_rows_tb_eigen_ADI)(FLT alpha, FLT * x, X(tb_eigen_ADI) * F);
void X(scale_columns_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F);
void X(scale_columns_tb_eigen_ADI)(FLT alpha, FLT * x, X(tb_eigen_ADI) * F);

void X(trmv)(char TRANS, int n, FLT * A, int LDA, FLT * x);
void X(trsv)(char TRANS, int n, FLT * A, int LDA, FLT * x);

void X(trmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N);
void X(trsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N);

void X(bfmv)(char TRANS, X(tb_eigen_FMM) * A, FLT * x);
void X(bfmv_ADI)(char TRANS, X(tb_eigen_ADI) * A, FLT * x);
void X(bfsv)(char TRANS, X(tb_eigen_FMM) * A, FLT * x);
void X(bfsv_ADI)(char TRANS, X(tb_eigen_ADI) * A, FLT * x);

void X(bfmm)(char TRANS, X(tb_eigen_FMM) * F, FLT * X, int LDX, int N);
void X(bfmm_ADI)(char TRANS, X(tb_eigen_ADI) * F, FLT * X, int LDX, int N);
void X(bfsm)(char TRANS, X(tb_eigen_FMM) * F, FLT * X, int LDX, int N);
void X(bfsm_ADI)(char TRANS, X(tb_eigen_ADI) * F, FLT * X, int LDX, int N);

FLT X(normest_tb_eigen_ADI)(X(tb_eigen_ADI) * F);

X(triangular_banded) * X(create_A_konoplev_to_jacobi)(const int n, const FLT alpha, const FLT beta);
X(triangular_banded) * X(create_B_konoplev_to_jacobi)(const int n, const FLT alpha);

X(banded) * X(create_jacobi_derivative)(const int norm, const int m, const int n, const int order, const FLT alpha, const FLT beta);
X(banded) * X(create_jacobi_multiplication)(const int norm, const int m, const int n, const FLT alpha, const FLT beta);
X(banded) * X(create_jacobi_raising)(const int norm, const int m, const int n, const FLT alpha, const FLT beta);
X(banded) * X(create_jacobi_lowering)(const int norm, const int m, const int n, const FLT alpha, const FLT beta);

X(banded) * X(create_laguerre_derivative)(const int norm, const int m, const int n, const int order, const FLT alpha);
X(banded) * X(create_laguerre_multiplication)(const int norm, const int m, const int n, const FLT alpha);
X(banded) * X(create_laguerre_raising)(const int norm, const int m, const int n, const FLT alpha);
X(banded) * X(create_laguerre_lowering)(const int norm, const int m, const int n, const FLT alpha);

X(banded) * X(create_hermite_derivative)(const int norm, const int m, const int n, const int order);
X(banded) * X(create_hermite_multiplication)(const int norm, const int m, const int n);

void X(create_associated_jacobi_to_jacobi_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT c, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta, FLT * D, const int INCD);
void X(create_associated_laguerre_to_laguerre_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT c, const FLT alpha, const FLT beta, FLT * D, const int INCD);
void X(create_associated_hermite_to_hermite_diagonal_connection_coefficient)(const int norm1, const int norm2, const int n, const FLT c, FLT * D, const int INCD);

X(triangular_banded) * X(create_A_associated_jacobi_to_jacobi)(const int norm, const int n, const int c, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta);
X(triangular_banded) * X(create_B_associated_jacobi_to_jacobi)(const int norm, const int n, const FLT gamma, const FLT delta);
X(triangular_banded) * X(create_C_associated_jacobi_to_jacobi)(const int norm, const int n, const FLT gamma, const FLT delta);

X(triangular_banded) * X(create_A_associated_laguerre_to_laguerre)(const int norm, const int n, const int c, const FLT alpha, const FLT beta);
X(triangular_banded) * X(create_B_associated_laguerre_to_laguerre)(const int norm, const int n, const FLT beta);
X(triangular_banded) * X(create_C_associated_laguerre_to_laguerre)(const int norm, const int n, const FLT beta);

X(triangular_banded) * X(create_A_associated_hermite_to_hermite)(const int norm, const int n, const int c);
X(triangular_banded) * X(create_B_associated_hermite_to_hermite)(const int norm, const int n);
X(triangular_banded) * X(create_C_associated_hermite_to_hermite)(const int n);

X(lowrankmatrix) * X(ddfadi)(const int m, const FLT * A, const int n, const FLT * B, const int b, const FLT * X, const FLT * Y);
