// FFTW overrides for synthesis and analysis on tensor product grids.

#include "fasttransforms.h"
#include "ftinternal.h"

void ft_destroy_sphere_fftw_plan(ft_sphere_fftw_plan * P) {
    fftw_destroy_plan(P->plantheta1);
    fftw_destroy_plan(P->plantheta2);
    fftw_destroy_plan(P->plantheta3);
    fftw_destroy_plan(P->plantheta4);
    fftw_destroy_plan(P->planphi);
    fftw_free(P->Y);
    free(P);
}

ft_sphere_fftw_plan * ft_plan_sph_synthesis(const int N, const int M) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_sphere_fftw_plan * P = malloc(sizeof(ft_sphere_fftw_plan));

    P->Y = fftw_malloc(N*M*sizeof(double));

    int howmany = (M+3)/4;
    fftw_r2r_kind kind[] = {FFTW_REDFT01};
    P->plantheta1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    kind[0] = FFTW_RODFT01;
    P->plantheta2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    kind[0] = FFTW_RODFT01;
    P->plantheta3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = M/4;
    kind[0] = FFTW_REDFT01;
    P->plantheta4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    kind[0] = FFTW_HC2R;
    P->planphi = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    return P;
}

ft_sphere_fftw_plan * ft_plan_sph_analysis(const int N, const int M) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_sphere_fftw_plan * P = malloc(sizeof(ft_sphere_fftw_plan));

    P->Y = fftw_malloc(N*M*sizeof(double));

    int howmany = (M+3)/4;
    fftw_r2r_kind kind[] = {FFTW_REDFT10};
    P->plantheta1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    kind[0] = FFTW_RODFT10;
    P->plantheta2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    kind[0] = FFTW_RODFT10;
    P->plantheta3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = M/4;
    kind[0] = FFTW_REDFT10;
    P->plantheta4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    kind[0] = FFTW_R2HC;
    P->planphi = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    return P;
}

void ft_execute_sph_synthesis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M) {
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
    colswap(X, P->Y, N, M);
    fftw_execute_r2r(P->planphi, P->Y, X);
}

void ft_execute_sph_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M) {
    fftw_execute_r2r(P->planphi, X, P->Y);
    colswap_t(X, P->Y, N, M);
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


void ft_destroy_triangle_fftw_plan(ft_triangle_fftw_plan * P) {
    fftw_destroy_plan(P->planxy);
    free(P);
}

ft_triangle_fftw_plan * ft_plan_tri_synthesis(const int N, const int M) {
    ft_triangle_fftw_plan * P = malloc(sizeof(ft_triangle_fftw_plan));
    double * X = fftw_malloc(N*M*sizeof(double));
    P->planxy = fftw_plan_r2r_2d(N, M, X, X, FFTW_REDFT01, FFTW_REDFT01, FT_FFTW_FLAGS);
    fftw_free(X);
    return P;
}

ft_triangle_fftw_plan * ft_plan_tri_analysis(const int N, const int M) {
    ft_triangle_fftw_plan * P = malloc(sizeof(ft_triangle_fftw_plan));
    double * X = fftw_malloc(N*M*sizeof(double));
    P->planxy = fftw_plan_r2r_2d(N, M, X, X, FFTW_REDFT10, FFTW_REDFT10, FT_FFTW_FLAGS);
    fftw_free(X);
    return P;
}

void ft_execute_tri_synthesis(const ft_triangle_fftw_plan * P, double * X, const int N, const int M) {
    if (N > 1 && M > 1) {
        for (int i = 0; i < N; i++)
            X[i] *= 2.0;
        for (int j = 0; j < M; j++)
            X[j*N] *= 2.0;
        fftw_execute_r2r(P->planxy, X, X);
        for (int i = 0; i < N*M; i++)
            X[i] *= 0.25;
    }
}

void ft_execute_tri_analysis(const ft_triangle_fftw_plan * P, double * X, const int N, const int M) {
    if (N > 1 && M > 1) {
        fftw_execute_r2r(P->planxy, X, X);
        for (int i = 0; i < N; i++)
            X[i] *= 0.5;
        for (int j = 0; j < M; j++)
            X[j*N] *= 0.5;
        for (int i = 0; i < N*M; i++)
            X[i] /= N*M;
    }
}


void ft_destroy_disk_fftw_plan(ft_disk_fftw_plan * P) {
    fftw_destroy_plan(P->planr1);
    fftw_destroy_plan(P->planr2);
    fftw_destroy_plan(P->planr3);
    fftw_destroy_plan(P->planr4);
    fftw_destroy_plan(P->plantheta);
    fftw_free(P->Y);
    free(P);
}

ft_disk_fftw_plan * ft_plan_disk_synthesis(const int N, const int M) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_disk_fftw_plan * P = malloc(sizeof(ft_disk_fftw_plan));

    P->Y = fftw_malloc(N*M*sizeof(double));

    int howmany = (M+3)/4;
    fftw_r2r_kind kind[] = {FFTW_REDFT01};
    P->planr1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    kind[0] = FFTW_REDFT11;
    P->planr2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    kind[0] = FFTW_REDFT11;
    P->planr3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = M/4;
    kind[0] = FFTW_REDFT01;
    P->planr4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    kind[0] = FFTW_HC2R;
    P->plantheta = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    return P;
}

ft_disk_fftw_plan * ft_plan_disk_analysis(const int N, const int M) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_disk_fftw_plan * P = malloc(sizeof(ft_disk_fftw_plan));

    P->Y = fftw_malloc(N*M*sizeof(double));

    int howmany = (M+3)/4;
    fftw_r2r_kind kind[] = {FFTW_REDFT10};
    P->planr1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    kind[0] = FFTW_REDFT11;
    P->planr2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    kind[0] = FFTW_REDFT11;
    P->planr3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    howmany = M/4;
    kind[0] = FFTW_REDFT10;
    P->planr4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    kind[0] = FFTW_R2HC;
    P->plantheta = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind, FT_FFTW_FLAGS);

    return P;
}

void ft_execute_disk_synthesis(const ft_disk_fftw_plan * P, double * X, const int N, const int M) {
    X[0] *= 2.0;
    for (int j = 3; j < M; j += 4) {
        X[j*N] *= 2.0;
        X[(j+1)*N] *= 2.0;
    }
    fftw_execute_r2r(P->planr1, X, X);
    fftw_execute_r2r(P->planr2, X+N, X+N);
    fftw_execute_r2r(P->planr3, X+2*N, X+2*N);
    fftw_execute_r2r(P->planr4, X+3*N, X+3*N);
    for (int i = 0; i < N*M; i++)
        X[i] *= M_1_4_SQRT_PI;
    for (int i = 0; i < N; i++)
        X[i] *= M_SQRT2;
    colswap(X, P->Y, N, M);
    fftw_execute_r2r(P->plantheta, P->Y, X);
}

void ft_execute_disk_analysis(const ft_disk_fftw_plan * P, double * X, const int N, const int M) {
    fftw_execute_r2r(P->plantheta, X, P->Y);
    colswap_t(X, P->Y, N, M);
    for (int i = 0; i < N*M; i++)
        X[i] *= M_4_SQRT_PI/(2*N*M);
    for (int i = 0; i < N; i++)
        X[i] *= M_SQRT1_2;
    fftw_execute_r2r(P->planr1, X, X);
    fftw_execute_r2r(P->planr2, X+N, X+N);
    fftw_execute_r2r(P->planr3, X+2*N, X+2*N);
    fftw_execute_r2r(P->planr4, X+3*N, X+3*N);
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
