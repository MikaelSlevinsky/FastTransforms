typedef struct {
    X(triangular_banded) * data[2][2];
    int n;
    int b;
} X(block_2x2_triangular_banded);

typedef struct {
    X(tb_eigen_FMM) * F;
    FLT * s;
    FLT * c;
    FLT * t;
    int n;
} X(btb_eigen_FMM);

void X(destroy_block_2x2_triangular_banded)(X(block_2x2_triangular_banded) * A);
void X(destroy_btb_eigen_FMM)(X(btb_eigen_FMM) * F);

X(block_2x2_triangular_banded) * X(create_block_2x2_triangular_banded)(X(triangular_banded) * data[2][2]);
X(triangular_banded) * X(convert_block_2x2_triangular_banded_to_triangular_banded)(X(block_2x2_triangular_banded) * A);

FLT X(get_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, const int i, const int j);
void X(set_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, const FLT v, const int i, const int j);
void X(block_get_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, FLT v[2][2], const int i, const int j);
void X(block_set_block_2x2_triangular_banded_index)(const X(block_2x2_triangular_banded) * A, const FLT v[2][2], const int i, const int j);

void X(btbmv)(char TRANS, X(block_2x2_triangular_banded) * A, FLT * x);
void X(btbsv)(char TRANS, X(block_2x2_triangular_banded) * A, FLT * x);

void X(block_2x2_triangular_banded_eigenvalues)(X(block_2x2_triangular_banded) * A, X(block_2x2_triangular_banded) * B, FLT * lambda);
void X(block_2x2_triangular_banded_eigenvectors)(X(block_2x2_triangular_banded) * A, X(block_2x2_triangular_banded) * B, FLT * V);

X(btb_eigen_FMM) * X(btb_eig_FMM)(X(block_2x2_triangular_banded) * A, X(block_2x2_triangular_banded) * B, FLT * D);

void X(btrmv)(char TRANS, int n, FLT * A, int LDA, FLT * x);
void X(btrsv)(char TRANS, int n, FLT * A, int LDA, FLT * x);

void X(btrmm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N);
void X(btrsm)(char TRANS, int n, FLT * A, int LDA, FLT * B, int LDB, int N);

void X(bbfmv)(char TRANS, X(btb_eigen_FMM) * F, FLT * x);
void X(bbfsv)(char TRANS, X(btb_eigen_FMM) * F, FLT * x);

void X(bbfmm)(char TRANS, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N);
void X(bbfsm)(char TRANS, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N);

void X(bbbfmv)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * x);
void X(bbbfsv)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * x);

void X(bbbfmm)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N);
void X(bbbfsm)(char TRANS, char DBLOCK, char RBLOCK, X(btb_eigen_FMM) * F, FLT * B, int LDB, int N);
