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

void X(destroy_banded)(X(banded) * A);
void X(destroy_triangular_banded)(X(triangular_banded) * A);
void X(destroy_tb_eigen_FMM)(X(tb_eigen_FMM) * F);

size_t X(summary_size_tb_eigen_FMM)(X(tb_eigen_FMM) * F);

X(banded) * X(malloc_banded)(const int m, const int n, const int l, const int u);
X(banded) * X(calloc_banded)(const int m, const int n, const int l, const int u);

X(triangular_banded) * X(malloc_triangular_banded)(const int n, const int b);
X(triangular_banded) * X(calloc_triangular_banded)(const int n, const int b);

FLT X(get_banded_index)(const X(banded) * A, const int i, const int j);
void X(set_banded_index)(const X(banded) * A, const FLT v, const int i, const int j);

FLT X(get_triangular_banded_index)(const X(triangular_banded) * A, const int i, const int j);
void X(set_triangular_banded_index)(const X(triangular_banded) * A, const FLT v, const int i, const int j);

void X(gbmv)(FLT alpha, X(banded) * A, FLT * x, FLT beta, FLT * y);
void X(gbmm)(FLT alpha, X(banded) * A, X(banded) * B, FLT beta, X(banded) * C);
void X(banded_add)(FLT alpha, X(banded) * A, FLT beta, X(banded) * B, X(banded) * C);

void X(tbmv)(char TRANS, X(triangular_banded) * A, FLT * x);
void X(tbsv)(char TRANS, X(triangular_banded) * A, FLT * x);

void X(triangular_banded_eigenvalues)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda);
void X(triangular_banded_eigenvectors)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * V);

void X(triangular_banded_eigenvalues_3arg)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda, X(triangular_banded) * C, FLT * omega);
void X(triangular_banded_eigenvectors_3arg)(X(triangular_banded) * A, X(triangular_banded) * B, FLT * lambda, X(triangular_banded) * C, FLT * V);

X(tb_eigen_FMM) * X(tb_eig_FMM)(X(triangular_banded) * A, X(triangular_banded) * B);

void X(scale_rows_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F);
void X(scale_columns_tb_eigen_FMM)(FLT alpha, FLT * x, X(tb_eigen_FMM) * F);

void X(trmv)(char TRANS, int n, FLT * A, int LDA, FLT * x);
void X(trsv)(char TRANS, int n, FLT * A, int LDA, FLT * x);

void X(trmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N);
void X(trsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N);

void X(bfmv)(char TRANS, X(tb_eigen_FMM) * A, FLT * x);
void X(bfsv)(char TRANS, X(tb_eigen_FMM) * A, FLT * x);

void X(bfmm)(char TRANS, X(tb_eigen_FMM) * F, FLT * X, int LDX, int N);
void X(bfsm)(char TRANS, X(tb_eigen_FMM) * F, FLT * X, int LDX, int N);

X(triangular_banded) * X(create_A_konoplev_to_jacobi)(const int n, const FLT alpha, const FLT beta);
X(triangular_banded) * X(create_B_konoplev_to_jacobi)(const int n, const FLT alpha);

X(banded) * X(create_jacobi_derivative)(const int m, const int n, const int order, const FLT alpha, const FLT beta);
X(banded) * X(create_jacobi_multiplication)(const int m, const int n, const FLT alpha, const FLT beta);
X(banded) * X(create_jacobi_raising)(const int m, const int n, const FLT alpha, const FLT beta);
X(banded) * X(create_jacobi_lowering)(const int m, const int n, const FLT alpha, const FLT beta);

X(triangular_banded) * X(create_A_associated_jacobi_to_jacobi)(const int n, const FLT alpha, const FLT beta, const FLT gamma, const FLT delta);
X(triangular_banded) * X(create_B_associated_jacobi_to_jacobi)(const int n, const FLT gamma, const FLT delta);
X(triangular_banded) * X(create_C_associated_jacobi_to_jacobi)(const int n, const FLT gamma, const FLT delta);
X(triangular_banded) * X(pre_ADI_Chebyshev_Legendre)(const int n, const FLT epsilon);

X(triangular_banded) * X(ADI_Chebyshev_Legendre)(const int n, const X(triangular_banded) * A, const X(triangular_banded) * B, const X(triangular_banded) * lambda, const X(triangular_banded) * V, const FLT epsilon);
                                                  
X(triangular_banded) * X(pre_ADI_Legendre_Legendre_first_associated)(const int n, const FLT epsilon);

X(triangular_banded) * X(ADI_Legendre_Legendre_first_associated)(const int n, X(triangular_banded)* A, X(triangular_banded)* B, X(triangular_banded)* W, X(triangular_banded)* lambda, const FLT epsilon);

X(densematrix) * X(ADI)(const X(triangular_banded) * A, const X(triangular_banded) * B, const X(densematrix) * F, const FLT epsilon);

X(densematrix) * X(tb_eigen_FMM_to_densematrix)(const X(tb_eigen_FMM) * F);

X(triangular_banded) * X(add_triangular_banded)(const X(triangular_banded) * A, const X(triangular_banded) * B, char pm);

X(triangular_banded) * X(shift_triangular_banded)(const X(triangular_banded) * A, const FLT alpha);

X(densematrix) * X(add_densematrix)(const X(densematrix) * A, const X(densematrix) * B, char pm);

FLT * X(eigenvalue_intervals)(const X(triangular_banded) * A, const X(triangular_banded) * B);

int X(number_of_shifts)(const FLT gamma, const FLT epsilon);

FLT X(determinant)(const int n, const FLT A[][n]);

X(triangular_banded) * X(triangular_banded_inverse)(const X(triangular_banded) * A);

FLT X(mobius)(const FLT z, const FLT a, const FLT b, const FLT c, const FLT d);

FLT X(geometric_arithmetic_mean)(const FLT a, const FLT b);

void X(Jacobi_elliptic_functions)(const FLT x, const FLT k, FLT * sn , FLT * cn, FLT * dn, FLT * dn1, const int n);

FLT ** X(ADI_shifts)(const X(triangular_banded) * A, const X(triangular_banded) * B, const FLT epsilon);

X(triangular_banded) ** X(block_divide)(const X(triangular_banded)* A, const int n, const int s);

X(densematrix) * X(get_A_12)(const X(triangular_banded)* A, const int n, const int s);

X(triangular_banded) * X(block_attach)(const X(triangular_banded) * A_11, const X(densematrix) * A_12, const X(triangular_banded) * A_22);

X(densematrix) * X(triangular_banded_multiply_densematrix)(const X(triangular_banded) * A, const X(densematrix) * B);

X(densematrix) * X(densematrix_multiply_triangular_banded)(const X(densematrix) * A, const X(triangular_banded) * B);

X(triangular_banded) * X(triangular_banded_multiply_triangular_banded)(const X(triangular_banded) * A, const X(triangular_banded) * B);

X(triangular_banded) * X(transpose_triangular_banded)(const X(triangular_banded) * A);

X(densematrix) * X(transpose_densematrix)(const X(densematrix) * A);
