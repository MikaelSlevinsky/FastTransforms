#include "fasttransforms.h"

void freeSphereFFTWPlan(SphereFFTWPlan * P) {
    fftw_destroy_plan(P->plantheta1);
    fftw_destroy_plan(P->plantheta2);
    fftw_destroy_plan(P->plantheta3);
    fftw_destroy_plan(P->plantheta4);
    fftw_destroy_plan(P->planphi);
    free(P->Y);
    free(P);
}

SphereFFTWPlan * plan_sph_synthesis(const int N, const int M) {
    int rank = 1; /* not 2: we are computing 1d transforms */
    int n[] = {N}; /* 1d transforms of length n */
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; /* distance between two elements in the same column */
    int * inembed = n, * onembed = n;

    SphereFFTWPlan * P = malloc(sizeof(SphereFFTWPlan));

    int howmany = (M+3)/4;
    fftw_r2r_kind kind[] = {FFTW_REDFT01};
    P->plantheta1 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    howmany = (M+2)/4;
    kind[0] = FFTW_RODFT01;
    P->plantheta2 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    howmany = (M+1)/4;
    kind[0] = FFTW_RODFT01;
    P->plantheta3 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    howmany = M/4;
    kind[0] = FFTW_REDFT01;
    P->plantheta4 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    kind[0] = FFTW_HC2R;
    P->planphi = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    P->Y = calloc(N*M, sizeof(double));
    return P;
}

SphereFFTWPlan * plan_sph_analysis(const int N, const int M) {
    int rank = 1; /* not 2: we are computing 1d transforms */
    int n[] = {N}; /* 1d transforms of length n */
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; /* distance between two elements in the same column */
    int * inembed = n, * onembed = n;

    SphereFFTWPlan * P = malloc(sizeof(SphereFFTWPlan));

    int howmany = (M+3)/4;
    fftw_r2r_kind kind[] = {FFTW_REDFT10};
    P->plantheta1 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    howmany = (M+2)/4;
    kind[0] = FFTW_RODFT10;
    P->plantheta2 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    howmany = (M+1)/4;
    kind[0] = FFTW_RODFT10;
    P->plantheta3 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    howmany = M/4;
    kind[0] = FFTW_REDFT10;
    P->plantheta4 = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    kind[0] = FFTW_R2HC;
    P->planphi = fftw_plan_many_r2r(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, kind, FFTW_ESTIMATE);

    P->Y = calloc(N*M, sizeof(double));
    return P;
}

void execute_sph_synthesis(const SphereFFTWPlan * P, double * X, const int N, const int M) {
    X[0] *= 2.0;
    for (int j = 3; j < M; j += 4) {
        X[j*N] *= 2.0;
        X[(j+1)*N] *= 2.0;
    }
    fftw_execute_r2r(P->plantheta1, X, X);
    fftw_execute_r2r(P->plantheta2, X+N, X+N);
    fftw_execute_r2r(P->plantheta3, X+2*N, X+2*N);
    fftw_execute_r2r(P->plantheta4, X+3*N, X+3*N);

    for (int i = 0; i < N*M; i++)
        X[i] *= M_1_4_SQRT_PI;

    for (int i = 0; i < N; i++)
        X[i] *= M_SQRT2;

    double * Y = P->Y;

    colswap(X, Y, N, M);

    fftw_execute_r2r(P->planphi, Y, X);
}

void execute_sph_analysis(const SphereFFTWPlan * P, double * X, const int N, const int M) {
    double * Y = P->Y;

    fftw_execute_r2r(P->planphi, X, Y);

    colswap_t(X, Y, N, M);

    for (int i = 0; i < N*M; i++)
        X[i] *= M_4_SQRT_PI/(2*N*M);

    for (int i = 0; i < N; i++)
        X[i] *= M_SQRT1_2;

    fftw_execute_r2r(P->plantheta1, X, X);
    fftw_execute_r2r(P->plantheta2, X+N, X+N);
    fftw_execute_r2r(P->plantheta3, X+2*N, X+2*N);
    fftw_execute_r2r(P->plantheta4, X+3*N, X+3*N);

    X[0] *= 0.5;
    for (int j = 3; j < M; j += 4) {
        X[j*N] *= 0.5;
        X[(j+1)*N] *= 0.5;
    }
}


static inline void colswap(const double * X, double * Y, const int N, const int M) {
    for (int i = 0; i < N; i++)
        Y[i] = X[i];
    for (int j = 1; j < (M+1)/2; j++) {
        for (int i = 0; i < N; i++)
            Y[i+j*N] = X[i+2*j*N];
        for (int i = 0; i < N; i++)
            Y[i+(M-j)*N] = -X[i+(2*j-1)*N];
    }
}

static inline void colswap_t(double * X, const double * Y, const int N, const int M) {
    for (int i = 0; i < N; i++)
        X[i] = Y[i];
    for (int j = 1; j < (M+1)/2; j++) {
        for (int i = 0; i < N; i++)
            X[i+2*j*N] = Y[i+j*N];
        for (int i = 0; i < N; i++)
            X[i+(2*j-1)*N] = -Y[i+(M-j)*N];
    }
}
