/*
Notes:
 - densematrix, lowrankmatrix are pointers to contiguous floating-point memory with sizes.
 - lowrankmatrix has two forms determined by the (integer) character constant N:
   when N == '2', L = UVᵀ, whereas when N == '3', L = USVᵀ. There are temporary
   arrays to allow fast linear algebra without further memory allocation.
 - a hierarchicalmatrix is self-referential (and therefore incompletely initialized).
   It stores pointers to hierarchical, dense, and low-rank matrices, and a hash table
   to determine which type of matrix is active in the current block under consideration.
   A hash value of 0 is provided with an initial `malloc`, which corresponds to
   initialized pointers (to pointers), but uninitialized data. As such,
   further initialization is required.
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
};

FLT * X(chebyshev_points)(char KIND, int n);
FLT * X(chebyshev_barycentric_weights)(char KIND, int n);

void X(destroy_densematrix)(X(densematrix) * A);
void X(destroy_lowrankmatrix)(X(lowrankmatrix) * A);
void X(destroy_hierarchicalmatrix)(X(hierarchicalmatrix) * A);

X(densematrix) * X(calloc_densematrix)(int m, int n);
X(densematrix) * X(malloc_densematrix)(int m, int n);
X(densematrix) * X(sample_densematrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);

X(lowrankmatrix) * X(calloc_lowrankmatrix)(char N, int m, int n, int r);
X(lowrankmatrix) * X(malloc_lowrankmatrix)(char N, int m, int n, int r);
X(lowrankmatrix) * X(sample_lowrankmatrix)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);

X(hierarchicalmatrix) * X(malloc_hierarchicalmatrix)(const int M, const int N);
X(hierarchicalmatrix) * X(create_hierarchicalmatrix)(const int M, const int N, FLT (*f)(FLT, FLT), const int m, const int n);

X(hierarchicalmatrix) * X(sample_hierarchicalmatrix) (FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);
X(hierarchicalmatrix) * X(sample_hierarchicalmatrix1)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);
X(hierarchicalmatrix) * X(sample_hierarchicalmatrix2)(FLT (*f)(FLT x, FLT y), FLT * x, FLT * y, unitrange i, unitrange j);

int X(size_densematrix)(X(densematrix) * A, int k);
int X(size_lowrankmatrix)(X(lowrankmatrix) * L, int k);
int X(size_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int k);
int X(blocksize_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int m, int n, int k);

FLT X(getindex_densematrix)(X(densematrix) * A, int i, int j);
FLT X(getindex_lowrankmatrix)(X(lowrankmatrix) * L, int i, int j);
FLT X(getindex_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int i, int j);
FLT X(blockgetindex_hierarchicalmatrix)(X(hierarchicalmatrix) * H, int m, int n, int i, int j);

void X(gemv)(char TRANS, int m, int n, FLT alpha, FLT * A, FLT * x, FLT beta, FLT * y);
void X(demv)(char TRANS, FLT alpha, X(densematrix) * A, FLT * x, FLT beta, FLT * y);
void X(lrmv)(char TRANS, FLT alpha, X(lowrankmatrix) * L, FLT * x, FLT beta, FLT * y);
void X(himv)(char TRANS, FLT alpha, X(hierarchicalmatrix) * H, FLT * x, FLT beta, FLT * y);

int X(binarysearch)(FLT * x, int start, int stop, FLT y);

void X(indsplit)(FLT * x, unitrange ir, unitrange * i1, unitrange * i2, FLT a, FLT b);

FLT X(cauchykernel)(FLT x, FLT y);
FLT X(coulombkernel)(FLT x, FLT y);
FLT X(coulombprimekernel)(FLT x, FLT y);
FLT X(logkernel)(FLT x, FLT y);
