typedef struct {
    FLT * a;
    FLT * b;
    int n;
} X(symmetric_tridiagonal);

// b is the superdiagonal vector, -b is the subdiagonal.
typedef struct {
    FLT * b;
    int n;
} X(skew_symmetric_tridiagonal);

typedef struct {
    FLT * c;
    FLT * d;
    int n;
} X(bidiagonal);

typedef struct {
    FLT * A;
    FLT * B;
    FLT * C;
    FLT * lambda;
    int sign;
    int n;
} X(symmetric_tridiagonal_symmetric_eigen);

void X(destroy_symmetric_tridiagonal)(X(symmetric_tridiagonal) * A);
void X(destroy_skew_symmetric_tridiagonal)(X(skew_symmetric_tridiagonal) * A);
void X(destroy_bidiagonal)(X(bidiagonal) * B);
void X(destroy_symmetric_tridiagonal_symmetric_eigen)(X(symmetric_tridiagonal_symmetric_eigen) * F);

X(bidiagonal) * X(symmetric_tridiagonal_cholesky)(X(symmetric_tridiagonal) * A);
void X(stmv)(char TRANS, FLT alpha, X(symmetric_tridiagonal) * A, FLT * x, FLT beta, FLT * y);
void X(ktmv)(char TRANS, FLT alpha, X(skew_symmetric_tridiagonal) * A, FLT * x, FLT beta, FLT * y);
void X(bdmv)(char TRANS, X(bidiagonal) * B, FLT * x);
void X(bdsv)(char TRANS, X(bidiagonal) * B, FLT * x);

void X(symmetric_tridiagonal_eig)(X(symmetric_tridiagonal) * A, FLT * V, FLT * lambda);
void X(symmetric_definite_tridiagonal_eig)(X(symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B, FLT * V, FLT * lambda);
X(symmetric_tridiagonal) * X(symmetric_tridiagonal_congruence)(X(symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B, FLT * V);

X(symmetric_tridiagonal_symmetric_eigen) * X(symmetric_tridiagonal_symmetric_eig)(X(symmetric_tridiagonal) * T, FLT * lambda, const int sign);
void X(semv)(X(symmetric_tridiagonal_symmetric_eigen) * F, FLT * x, int incx, FLT * y);

void X(skew_to_symmetric_tridiagonal)(X(skew_symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B, X(symmetric_tridiagonal) * C);

X(symmetric_tridiagonal) * X(create_A_shtsdtev)(const int n, const int mu, const int m, char PARITY);
X(symmetric_tridiagonal) * X(create_B_shtsdtev)(const int n, const int m, char PARITY);
X(bidiagonal) * X(create_R_shtsdtev)(const int n, const int m, char PARITY);

X(skew_symmetric_tridiagonal) * X(create_rectdisk_angular_momentum)(const int n, const FLT beta);
