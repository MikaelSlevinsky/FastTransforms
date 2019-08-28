/*
Notes:
 - densematrix, lowrankmatrix are pointers to contiguous floating-point memory with sizes.
 - lowrankmatrix has two forms determined by the (integer) character constant N:
   when N == '2', L = UVᵀ, whereas when N == '3', L = USVᵀ. There are temporary
   arrays to allow fast linear algebra without further memory allocation.
 - hierarchicalmatrix is self-referential (and therefore incompletely initialized).
   It stores pointers to hierarchical, dense, and low-rank matrices, and a hash table
   to determine which type of matrix is active in the current block under consideration.
   A hash value of 0 is provided with an initial `malloc`, which corresponds to
   initialized pointers (to pointers), but uninitialized data. As such,
   further initialization is required.
 - char SPLITTING controls the partitioning of unitranges for sampling a hierarchicalmatrix.
   SPLITTING == 'I' partitions unitranges by index,
   SPLITTING == 'G' partitions unitranges by geometry of associated samples.
*/

typedef struct {
    FLT * A;
    int m;
    int n;
} X(densematrix);

typedef struct {
    FLT * U;
    FLT * S;
    FLT * V;
    FLT * t1;
    FLT * t2;
    int m;
    int n;
    int r;
    int p;
    char N;
} X(lowrankmatrix);

typedef struct X(hmat) X(hierarchicalmatrix);

struct X(hmat) {
    X(hierarchicalmatrix) ** hierarchicalmatrices;
    X(densematrix) ** densematrices;
    X(lowrankmatrix) ** lowrankmatrices;
    int * hash;
    int M;
    int N;
    int m;
    int n;
};

FLT * X(chebyshev_points)(char KIND, int n);
FLT * X(chebyshev_barycentric_weights)(char KIND, int n);
void * X(barycentricmatrix)(FLT * A, FLT * x, int m, FLT * y, FLT * l, int n);

void X(destroy_densematrix)(X(densematrix) * A);
void X(destroy_lowrankmatrix)(X(lowrankmatrix) * A);
void X(destroy_hierarchicalmatrix)(X(hierarchicalmatrix) * A);

X(densematrix) * X(calloc_densematrix)(int m, int n);
X(densematrix) * X(malloc_densematrix)(int m, int n);
X(densematrix) * X(sample_densematrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);
X(densematrix) * X(sample_accurately_densematrix)(FLT (*f)(FLT x, FLT ylo, FLT yhi), FLT * x, FLT * ylo, FLT * yhi, unitrange i, unitrange j);

X(lowrankmatrix) * X(calloc_lowrankmatrix)(char N, int m, int n, int r);
X(lowrankmatrix) * X(malloc_lowrankmatrix)(char N, int m, int n, int r);
X(lowrankmatrix) * X(sample_lowrankmatrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);

X(hierarchicalmatrix) * X(malloc_hierarchicalmatrix)(const int M, const int N);
X(hierarchicalmatrix) * X(sample_hierarchicalmatrix) (FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j, char SPLITTING);
X(hierarchicalmatrix) * X(sample_accurately_hierarchicalmatrix) (FLT (*f)(FLT x, FLT y), FLT (*f2)(FLT x, FLT ylo, FLT yhi), FLT * x, FLT * y, FLT * ylo, FLT * yhi, unitrange i, unitrange j, char SPLITTING);

size_t X(summary_size_densematrix)(X(densematrix) * A);
size_t X(summary_size_lowrankmatrix)(X(lowrankmatrix) * L);
size_t X(summary_size_hierarchicalmatrix)(X(hierarchicalmatrix) * H);

int X(nlevels_hierarchicalmatrix)(X(hierarchicalmatrix) * H);

FLT X(norm_densematrix)(X(densematrix) * A);
FLT X(norm_lowrankmatrix)(X(lowrankmatrix) * L);
FLT X(norm_hierarchicalmatrix)(X(hierarchicalmatrix) * H);

FLT X(getindex_densematrix)(X(densematrix) * A, int i, int j);
FLT X(getindex_lowrankmatrix)(X(lowrankmatrix) * L, int i, int j);
FLT X(getindex_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int i, int j);
FLT X(blockgetindex_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int m, int n, int i, int j);

void X(scale_rows_densematrix)(FLT alpha, FLT * x, X(densematrix) * AD);
void X(scale_columns_densematrix)(FLT alpha, FLT * x, X(densematrix) * AD);
void X(scale_rows_lowrankmatrix)(FLT alpha, FLT * x, X(lowrankmatrix) * L);
void X(scale_columns_lowrankmatrix)(FLT alpha, FLT * x, X(lowrankmatrix) * L);
void X(scale_rows_hierarchicalmatrix)(FLT alpha, FLT * x, X(hierarchicalmatrix) * H);
void X(scale_columns_hierarchicalmatrix)(FLT alpha, FLT * x, X(hierarchicalmatrix) * H);

void X(gemv)(char TRANS, int m, int n, FLT alpha, FLT * A, int LDA, FLT * x, FLT beta, FLT * y);
void X(gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT * A, int LDA, FLT * B, int LDB, FLT beta, FLT * C, int LDC);
void X(demv)(char TRANS, FLT alpha, X(densematrix) * A, FLT * x, FLT beta, FLT * y);
void X(demm)(char TRANS, int p, FLT alpha, X(densematrix) * A, FLT * B, int LDB, FLT beta, FLT * C, int LDC);
void X(lrmv)(char TRANS, FLT alpha, X(lowrankmatrix) * L, FLT * x, FLT beta, FLT * y);
void X(lrmm)(char TRANS, int p, FLT alpha, X(lowrankmatrix) * L, FLT * B, int LDB, FLT beta, FLT * C, int LDC);
void X(ghmv)(char TRANS, FLT alpha, X(hierarchicalmatrix) * H, FLT * x, FLT beta, FLT * y);
void X(ghmm)(char TRANS, int p, FLT alpha, X(hierarchicalmatrix) * H, FLT * B, int LDB, FLT beta, FLT * C, int LDC);

int X(binarysearch)(FLT * x, int start, int stop, FLT y);

void X(indsplit)(FLT * x, unitrange ir, unitrange * i1, unitrange * i2, FLT a, FLT b);

FLT X(cauchykernel)(FLT x, FLT y);
FLT X(coulombkernel)(FLT x, FLT y);
FLT X(coulombprimekernel)(FLT x, FLT y);
FLT X(logkernel)(FLT x, FLT y);

static FLT X(diff)(FLT x, FLT y);

FLT X(cauchykernel2)(FLT x, FLT ylo, FLT yhi);
FLT X(coulombkernel2)(FLT x, FLT ylo, FLT yhi);
FLT X(coulombprimekernel2)(FLT x, FLT ylo, FLT yhi);
FLT X(logkernel2)(FLT x, FLT ylo, FLT yhi);
