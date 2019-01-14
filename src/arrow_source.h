typedef struct {
    FLT * a;
    FLT * b;
    FLT c;
    int n;
} X(symmetric_arrow);

typedef struct {
    FLT * d;
    FLT * e;
    FLT f;
    int n;
} X(upper_arrow);

typedef struct {
    FLT * Q;
    FLT * lambda;
    int * p;
    int n;
} X(symmetric_arrow_eigen);

typedef struct {
    X(hierarchicalmatrix) * Q;
    FLT * q;
    FLT * lambda;
    int * p;
    int n;
    int ib;
} X(symmetric_arrow_eigen_FMM);


void X(destroy_symmetric_arrow)(X(symmetric_arrow) * A);
void X(destroy_upper_arrow)(X(upper_arrow) * R);
void X(destroy_symmetric_arrow_eigen)(X(symmetric_arrow_eigen) * F);
void X(destroy_symmetric_arrow_eigen_FMM)(X(symmetric_arrow_eigen_FMM) * F);

X(upper_arrow) * X(symmetric_arrow_cholesky)(X(symmetric_arrow) * A);
X(upper_arrow) * X(upper_arrow_inv)(X(upper_arrow) * R);
X(symmetric_arrow) * X(symmetric_arrow_similarity)(X(symmetric_arrow) * A, X(symmetric_arrow) * B);
X(symmetric_arrow) * X(symmetric_arrow_synthesize)(X(symmetric_arrow) * A, FLT * lambda);

void X(samv)(char TRANS, FLT alpha, X(symmetric_arrow) * A, FLT * x, FLT beta, FLT * y);
void X(uamv)(char TRANS, X(upper_arrow) * R, FLT * x);
void X(uasv)(char TRANS, X(upper_arrow) * R, FLT * x);

FLT X(secular)(X(symmetric_arrow) * A, FLT lambda);
FLT * X(secular_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib);
FLT X(secular_derivative)(X(symmetric_arrow) * A, FLT lambda);
FLT * X(secular_derivative_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib);
FLT X(secular_second_derivative)(X(symmetric_arrow) * A, FLT lambda);
FLT * X(secular_second_derivative_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib);

FLT X(first_initial_guess)(FLT a0, FLT nrmb2, FLT c);
FLT X(first_pick_zero_update)(X(symmetric_arrow) * A, FLT lambda, int ib);
FLT X(last_initial_guess)(FLT an, FLT nrmb2, FLT c);
FLT X(last_pick_zero_update)(X(symmetric_arrow) * A, FLT lambda);
FLT X(pick_zero_update)(X(symmetric_arrow) * A, FLT lambda, int j);
FLT * X(pick_zero_update_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib);

int X(symmetric_arrow_deflate)(X(symmetric_arrow) * A, int * p);

FLT * X(symmetric_arrow_eigvals)(X(symmetric_arrow) * A, int ib);
FLT * X(symmetric_arrow_eigvals_FMM)(X(symmetric_arrow) * A, int ib);
FLT * X(symmetric_arrow_eigvecs)(X(symmetric_arrow) * A, FLT * lambda, int ib);
X(symmetric_arrow_eigen) * X(symmetric_arrow_eig)(X(symmetric_arrow) * A);
X(symmetric_arrow_eigen_FMM) * X(symmetric_arrow_eig_FMM)(X(symmetric_arrow) * A);

void X(quicksort)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
int X(partition)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y));
FLT X(selectpivot)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y));

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
