typedef struct X(tdcstruct) X(tdc_eigen);

struct X(tdcstruct) {
    X(symmetric_dpr1_eigen) * F0;
    X(tdc_eigen) * F1;
    X(tdc_eigen) * F2;
    FLT * V;
    FLT * lambda;
    int n;
};

typedef struct X(tdcstruct_FMM) X(tdc_eigen_FMM);

struct X(tdcstruct_FMM) {
    X(symmetric_dpr1_eigen_FMM) * F0;
    X(tdc_eigen_FMM) * F1;
    X(tdc_eigen_FMM) * F2;
    FLT * V;
    FLT * lambda;
    int n;
};

void X(destroy_tdc_eigen)(X(tdc_eigen) * F);
void X(destroy_tdc_eigen_FMM)(X(tdc_eigen_FMM) * F);

X(tdc_eigen) * X(tdc_eig)(X(symmetric_tridiagonal) * A);
X(tdc_eigen) * X(sdtdc_eig)(X(symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B);

X(tdc_eigen_FMM) * X(tdc_eig_FMM)(X(symmetric_tridiagonal) * A);
X(tdc_eigen_FMM) * X(sdtdc_eig_FMM)(X(symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B);

void X(tdmv)(char TRANS, FLT alpha, X(tdc_eigen) * F, FLT * x, FLT beta, FLT * y);
void X(tfmv)(char TRANS, FLT alpha, X(tdc_eigen_FMM) * F, FLT * x, FLT beta, FLT * y);
