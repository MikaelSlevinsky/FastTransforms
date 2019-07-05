/*
Notes:
 - symmetric_dpr1  == D+ρzzᵀ
 - symmetric_idpr1 == I+σzzᵀ
*/

typedef struct {
    FLT * d;
    FLT * z;
    FLT rho;
    int n;
} X(symmetric_dpr1);

typedef struct {
    FLT * z;
    FLT sigma;
    int n;
} X(symmetric_idpr1);

typedef struct {
    FLT * v;
    FLT * V;
    FLT * lambda;
    FLT * lambdalo;
    FLT * lambdahi;
    int * p;
    int * q;
    int n;
    int iz;
    int id;
} X(symmetric_dpr1_eigen);

typedef struct {
    FLT * v;
    X(hierarchicalmatrix) * V;
    FLT * lambda;
    FLT * lambdalo;
    FLT * lambdahi;
    int * p;
    int * q;
    int n;
    int iz;
    int id;
} X(symmetric_dpr1_eigen_FMM);


void X(destroy_symmetric_dpr1)(X(symmetric_dpr1) * A);
void X(destroy_symmetric_idpr1)(X(symmetric_idpr1) * A);
void X(destroy_symmetric_dpr1_eigen)(X(symmetric_dpr1_eigen) * F);
void X(destroy_symmetric_dpr1_eigen_FMM)(X(symmetric_dpr1_eigen_FMM) * F);

X(symmetric_idpr1) * X(symmetric_idpr1_factorize)(X(symmetric_idpr1) * A);
X(symmetric_idpr1) * X(symmetric_idpr1_inv)(X(symmetric_idpr1) * A);
X(symmetric_dpr1) * X(symmetric_dpr1_inv)(X(symmetric_dpr1) * A);
void X(symmetric_dpr1_synthesize)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi);
void X(symmetric_definite_dpr1_synthesize)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi);

void X(drmv)(char TRANS, X(symmetric_dpr1) * A, FLT * x);
void X(irmv)(char TRANS, X(symmetric_idpr1) * A, FLT * x);
void X(dvmv)(char TRANS, FLT alpha, X(symmetric_dpr1_eigen) * F, FLT * x, FLT beta, FLT * y);
void X(dfmv)(char TRANS, FLT alpha, X(symmetric_dpr1_eigen_FMM) * F, FLT * x, FLT beta, FLT * y);

FLT X(secular)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi);
FLT X(secular_derivative)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi);
FLT X(secular_second_derivative)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi);

FLT X(generalized_secular)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi);
FLT X(generalized_secular_derivative)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi);
FLT X(generalized_secular_second_derivative)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi);

FLT X(exterior_initial_guess)(FLT d0n, FLT nrmz2, FLT rho);
FLT X(first_pick_zero_update)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi);
FLT X(first_generalized_pick_zero_update)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi);
FLT X(last_pick_zero_update)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi);
FLT X(last_generalized_pick_zero_update)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi);
FLT X(pick_zero_update)(X(symmetric_dpr1) * A, FLT x0, FLT x1, FLT lambdalo, FLT lambdahi);
FLT X(generalized_pick_zero_update)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT x0, FLT x1, FLT lambdalo, FLT lambdahi);

void X(pick_zero_update_FMM)(X(symmetric_dpr1) * A, FLT * b2, FLT * lambda, FLT * delta, FLT * f, FLT * fp, FLT * fpp, int ib);

void X(symmetric_dpr1_deflate)(X(symmetric_dpr1) * A, int * p);
int X(symmetric_dpr1_deflate2)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi, int * p);
void X(symmetric_definite_dpr1_deflate)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, int * p);
int X(symmetric_definite_dpr1_deflate2)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi, int * p);

void X(symmetric_dpr1_eigvals)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi);
void X(symmetric_definite_dpr1_eigvals)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi);
FLT * X(symmetric_dpr1_eigvecs)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi, int m);
X(hierarchicalmatrix) * X(symmetric_dpr1_eigvecs_FMM)(X(symmetric_dpr1) * A, FLT * lambda, FLT * lambdalo, FLT * lambdahi, int m);
FLT * X(symmetric_definite_dpr1_eigvecs)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi, int m);
X(hierarchicalmatrix) * X(symmetric_definite_dpr1_eigvecs_FMM)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambda, FLT * lambdalo, FLT * lambdahi, int m);

X(symmetric_dpr1_eigen) * X(symmetric_dpr1_eig)(X(symmetric_dpr1) * A);
X(symmetric_dpr1_eigen_FMM) * X(symmetric_dpr1_eig_FMM)(X(symmetric_dpr1) * A);
X(symmetric_dpr1_eigen) * X(symmetric_definite_dpr1_eig)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B);
X(symmetric_dpr1_eigen_FMM) * X(symmetric_definite_dpr1_eig_FMM)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B);

void X(perm)(char DIRECTION, FLT * x, int * p, int n);

void X(quicksort_1arg)(FLT * a, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
int X(partition_1arg)(FLT * a, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
FLT X(selectpivot_1arg)(FLT * a, int * p, int lo, int hi, int (*by)(FLT x, FLT y));

void X(quicksort_2arg)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
int X(partition_2arg)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
FLT X(selectpivot_2arg)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y));

void X(quicksort_3arg)(FLT * a, FLT * b, FLT * c, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
int X(partition_3arg)(FLT * a, FLT * b, FLT * c, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
FLT X(selectpivot_3arg)(FLT * a, FLT * b, FLT * c, int * p, int lo, int hi, int (*by)(FLT x, FLT y));

void X(quicksort_4arg)(FLT * a, FLT * b, FLT * c, FLT * d, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
int X(partition_4arg)(FLT * a, FLT * b, FLT * c, FLT * d, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
FLT X(selectpivot_4arg)(FLT * a, FLT * b, FLT * c, FLT * d, int * p, int lo, int hi, int (*by)(FLT x, FLT y));

void X(swap)(FLT * a, int i, int j);
void X(swapi)(int * p, int i, int j);

int X(lt)(FLT x, FLT y);
int X(le)(FLT x, FLT y);
int X(gt)(FLT x, FLT y);
int X(ge)(FLT x, FLT y);
int X(ltabs)(FLT x, FLT y);
int X(leabs)(FLT x, FLT y);
int X(gtabs)(FLT x, FLT y);
int X(geabs)(FLT x, FLT y);
