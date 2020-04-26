// FFTW overrides for synthesis and analysis on tensor product grids.

#include "fasttransforms.h"
#include "ftinternal.h"

static inline void colswap(const ft_complex * X, ft_complex * Y, const int N, const int M) {
    for (int i = 0; i < N; i++) {
        Y[i][0] = X[i][0];
        Y[i][1] = X[i][1];
    }
    for (int j = 1; j < (M+1)/2; j++) {
        for (int i = 0; i < N; i++) {
            Y[i+j*N][0] = X[i+2*j*N][0];
            Y[i+j*N][1] = X[i+2*j*N][1];
        }
        for (int i = 0; i < N; i++) {
            Y[i+(M-j)*N][0] = X[i+(2*j-1)*N][0];
            Y[i+(M-j)*N][1] = X[i+(2*j-1)*N][1];
        }
    }
}

static inline void colswap_t(ft_complex * X, const ft_complex * Y, const int N, const int M) {
    for (int i = 0; i < N; i++) {
        X[i][0] = Y[i][0];
        X[i][1] = Y[i][1];
    }
    for (int j = 1; j < (M+1)/2; j++) {
        for (int i = 0; i < N; i++) {
            X[i+2*j*N][0] = Y[i+j*N][0];
            X[i+2*j*N][1] = Y[i+j*N][1];
        }
        for (int i = 0; i < N; i++) {
            X[i+(2*j-1)*N][0] = Y[i+(M-j)*N][0];
            X[i+(2*j-1)*N][1] = Y[i+(M-j)*N][1];
        }
    }
}

static inline void data_r2c(const double * X, double * Y, const int N, const int M) {
    for (int i = 0; i < N; i++)
        Y[2*i] = X[i];
    for (int j = 1; j < M/2+1; j++) {
        for (int i = 0; i < N; i++)
            Y[2*i+2*j*N] = X[i+(2*j)*N];
        for (int i = 0; i < N; i++)
            Y[2*i+1+2*j*N] = -X[i+(2*j-1)*N];
    }
}

static inline void data_c2r(double * X, const double * Y, const int N, const int M) {
    for (int i = 0; i < N; i++)
        X[i] = Y[2*i];
    for (int j = 1; j < M/2+1; j++) {
        for (int i = 0; i < N; i++)
            X[i+(2*j)*N] = Y[2*i+2*j*N];
        for (int i = 0; i < N; i++)
            X[i+(2*j-1)*N] = -Y[2*i+1+2*j*N];
    }
}

int ft_fftw_init_threads(void) {return fftw_init_threads();}
void ft_fftw_plan_with_nthreads(const int n) {return fftw_plan_with_nthreads(n);}

void ft_destroy_sphere_fftw_plan(ft_sphere_fftw_plan * P) {
    fftw_destroy_plan(P->plantheta1);
    fftw_destroy_plan(P->plantheta2);
    fftw_destroy_plan(P->plantheta3);
    fftw_destroy_plan(P->plantheta4);
    fftw_destroy_plan(P->planphi);
    fftw_free(P->Y);
    free(P);
}

ft_sphere_fftw_plan * ft_plan_sph_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1]) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_sphere_fftw_plan * P = malloc(sizeof(ft_sphere_fftw_plan));

    P->Y = fftw_malloc(N*2*(M/2+1)*sizeof(double));

    int howmany = (M+3)/4;
    P->plantheta1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[0], FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    P->plantheta2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[1], FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    P->plantheta3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[1], FT_FFTW_FLAGS);

    howmany = M/4;
    P->plantheta4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[0], FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    if (kind[2][0] == FFTW_HC2R)
        P->planphi = fftw_plan_many_dft_c2r(rank, n, howmany, (fftw_complex *) P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, FT_FFTW_FLAGS);
    else if (kind[2][0] == FFTW_R2HC)
        P->planphi = fftw_plan_many_dft_r2c(rank, n, howmany, P->Y, inembed, istride, idist, (fftw_complex *) P->Y, onembed, ostride, odist, FT_FFTW_FLAGS);
    return P;
}

ft_sphere_fftw_plan * ft_plan_sph_synthesis(const int N, const int M) {
    const fftw_r2r_kind kind[3][1] = {{FFTW_REDFT01}, {FFTW_RODFT01}, {FFTW_HC2R}};
    return ft_plan_sph_with_kind(N, M, kind);
}

ft_sphere_fftw_plan * ft_plan_sph_analysis(const int N, const int M) {
    const fftw_r2r_kind kind[3][1] = {{FFTW_REDFT10}, {FFTW_RODFT10}, {FFTW_R2HC}};
    return ft_plan_sph_with_kind(N, M, kind);
}

ft_sphere_fftw_plan * ft_plan_sphv_synthesis(const int N, const int M) {
    const fftw_r2r_kind kind[3][1] = {{FFTW_RODFT01}, {FFTW_REDFT01}, {FFTW_HC2R}};
    return ft_plan_sph_with_kind(N, M, kind);
}

ft_sphere_fftw_plan * ft_plan_sphv_analysis(const int N, const int M) {
    const fftw_r2r_kind kind[3][1] = {{FFTW_RODFT10}, {FFTW_REDFT10}, {FFTW_R2HC}};
    return ft_plan_sph_with_kind(N, M, kind);
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
    data_r2c(X, P->Y, N, M);
    fftw_execute_dft_c2r(P->planphi, (fftw_complex *) P->Y, X);
}

void ft_execute_sph_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M) {
    fftw_execute_dft_r2c(P->planphi, X, (fftw_complex *) P->Y);
    data_c2r(X, P->Y, N, M);
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

void ft_execute_sphv_synthesis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M) {
    for (int j = 1; j < M-2; j += 4) {
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
    data_r2c(X, P->Y, N, M);
    fftw_execute_dft_c2r(P->planphi, (fftw_complex *) P->Y, X);
}

void ft_execute_sphv_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M) {
    fftw_execute_dft_r2c(P->planphi, X, (fftw_complex *) P->Y);
    data_c2r(X, P->Y, N, M);
    for (int i = 0; i < N*M; i++)
        X[i] *= M_4_SQRT_PI/(2*N*M);
    for (int i = 0; i < N; i++)
        X[i] *= M_SQRT1_2;
    fftw_execute_r2r(P->plantheta1, X, X);
    fftw_execute_r2r(P->plantheta2, X+N, X+N);
    fftw_execute_r2r(P->plantheta3, X+2*N, X+2*N);
    fftw_execute_r2r(P->plantheta4, X+3*N, X+3*N);
    for (int j = 1; j < M-2; j += 4) {
        X[j*N] *= 0.5;
        X[(j+1)*N] *= 0.5;
    }
}


void ft_destroy_triangle_fftw_plan(ft_triangle_fftw_plan * P) {
    fftw_destroy_plan(P->planxy);
    free(P);
}

ft_triangle_fftw_plan * ft_plan_tri_with_kind(const int N, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1) {
    ft_triangle_fftw_plan * P = malloc(sizeof(ft_triangle_fftw_plan));
    double * X = fftw_malloc(N*M*sizeof(double));
    P->planxy = fftw_plan_r2r_2d(N, M, X, X, kind0, kind1, FT_FFTW_FLAGS);
    fftw_free(X);
    return P;
}

ft_triangle_fftw_plan * ft_plan_tri_synthesis(const int N, const int M) {return ft_plan_tri_with_kind(N, M, FFTW_REDFT01, FFTW_REDFT01);}
ft_triangle_fftw_plan * ft_plan_tri_analysis(const int N, const int M) {return ft_plan_tri_with_kind(N, M, FFTW_REDFT10, FFTW_REDFT10);}

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


void ft_destroy_tetrahedron_fftw_plan(ft_tetrahedron_fftw_plan * P) {
    fftw_destroy_plan(P->planxyz);
    free(P);
}

ft_tetrahedron_fftw_plan * ft_plan_tet_with_kind(const int N, const int L, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1, const fftw_r2r_kind kind2) {
    ft_tetrahedron_fftw_plan * P = malloc(sizeof(ft_tetrahedron_fftw_plan));
    double * X = fftw_malloc(N*L*M*sizeof(double));
    P->planxyz = fftw_plan_r2r_3d(N, L, M, X, X, kind0, kind1, kind2, FT_FFTW_FLAGS);
    fftw_free(X);
    return P;
}

ft_tetrahedron_fftw_plan * ft_plan_tet_synthesis(const int N, const int L, const int M) {return ft_plan_tet_with_kind(N, L, M, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01);}
ft_tetrahedron_fftw_plan * ft_plan_tet_analysis(const int N, const int L, const int M) {return ft_plan_tet_with_kind(N, L, M, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10);}

void ft_execute_tet_synthesis(const ft_tetrahedron_fftw_plan * P, double * X, const int N, const int L, const int M) {
    if (N > 1 && L > 1 && M > 1) {
        for (int j = 0; j < L; j++)
            for (int i = 0; i < N; i++)
                X[i+j*N] *= 2.0;
        for (int k = 0; k < M; k++)
            for (int j = 0; j < L; j++)
                X[(j+k*L)*N] *= 2.0;
        for (int k = 0; k < M; k++)
            for (int i = 0; i < N; i++)
                X[i+k*L*N] *= 2.0;
        fftw_execute_r2r(P->planxyz, X, X);
        for (int i = 0; i < N*L*M; i++)
            X[i] *= 0.125;
    }
}

void ft_execute_tet_analysis(const ft_tetrahedron_fftw_plan * P, double * X, const int N, const int L, const int M) {
    if (N > 1 && L > 1 && M > 1) {
        fftw_execute_r2r(P->planxyz, X, X);
        for (int j = 0; j < L; j++)
            for (int i = 0; i < N; i++)
                X[i+j*N] *= 0.5;
        for (int k = 0; k < M; k++)
            for (int j = 0; j < L; j++)
                X[(j+k*L)*N] *= 0.5;
        for (int k = 0; k < M; k++)
            for (int i = 0; i < N; i++)
                X[i+k*L*N] *= 0.5;
        for (int i = 0; i < N*L*M; i++)
            X[i] /= N*L*M;
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

ft_disk_fftw_plan * ft_plan_disk_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1]) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 4*N, odist = 4*N;
    int istride = 1, ostride = 1; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_disk_fftw_plan * P = malloc(sizeof(ft_disk_fftw_plan));

    P->Y = fftw_malloc(N*2*(M/2+1)*sizeof(double));

    int howmany = (M+3)/4;
    P->planr1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[0], FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    P->planr2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[1], FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    P->planr3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[1], FT_FFTW_FLAGS);

    howmany = M/4;
    P->planr4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[0], FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    if (kind[2][0] == FFTW_HC2R)
        P->plantheta = fftw_plan_many_dft_c2r(rank, n, howmany, (fftw_complex *) P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, FT_FFTW_FLAGS);
    else if (kind[2][0] == FFTW_R2HC)
        P->plantheta = fftw_plan_many_dft_r2c(rank, n, howmany, P->Y, inembed, istride, idist, (fftw_complex *) P->Y, onembed, ostride, odist, FT_FFTW_FLAGS);
    return P;
}

ft_disk_fftw_plan * ft_plan_disk_synthesis(const int N, const int M) {
    const fftw_r2r_kind kind[3][1] = {{FFTW_REDFT01}, {FFTW_REDFT11}, {FFTW_HC2R}};
    return ft_plan_disk_with_kind(N, M, kind);
}

ft_disk_fftw_plan * ft_plan_disk_analysis(const int N, const int M) {
    const fftw_r2r_kind kind[3][1] = {{FFTW_REDFT10}, {FFTW_REDFT11}, {FFTW_R2HC}};
    return ft_plan_disk_with_kind(N, M, kind);
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
    data_r2c(X, P->Y, N, M);
    fftw_execute_dft_c2r(P->plantheta, (fftw_complex *) P->Y, X);
}

void ft_execute_disk_analysis(const ft_disk_fftw_plan * P, double * X, const int N, const int M) {
    fftw_execute_dft_r2c(P->plantheta, X, (fftw_complex *) P->Y);
    data_c2r(X, P->Y, N, M);
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


void ft_destroy_spinsphere_fftw_plan(ft_spinsphere_fftw_plan * P) {
    fftw_destroy_plan(P->plantheta1);
    fftw_destroy_plan(P->plantheta2);
    fftw_destroy_plan(P->plantheta3);
    fftw_destroy_plan(P->plantheta4);
    fftw_destroy_plan(P->planphi);
    fftw_free(P->Y);
    free(P);
}

ft_spinsphere_fftw_plan * ft_plan_spinsph_with_kind(const int N, const int M, const int S, const fftw_r2r_kind kind[2][1], const int sign) {
    int rank = 1; // not 2: we are computing 1d transforms //
    int n[] = {N}; // 1d transforms of length n //
    int idist = 8*N, odist = 8*N;
    int istride = 2, ostride = 2; // distance between two elements in the same column //
    int * inembed = n, * onembed = n;

    ft_spinsphere_fftw_plan * P = malloc(sizeof(ft_spinsphere_fftw_plan));

    P->Y = fftw_malloc(2*N*M*sizeof(double));

    int howmany = (M+3)/4;
    P->plantheta1 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[0], FT_FFTW_FLAGS);

    howmany = (M+2)/4;
    P->plantheta2 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[1], FT_FFTW_FLAGS);

    howmany = (M+1)/4;
    P->plantheta3 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[1], FT_FFTW_FLAGS);

    howmany = M/4;
    P->plantheta4 = fftw_plan_many_r2r(rank, n, howmany, P->Y, inembed, istride, idist, P->Y, onembed, ostride, odist, kind[0], FT_FFTW_FLAGS);

    n[0] = M;
    idist = odist = 1;
    istride = ostride = N;
    howmany = N;
    P->planphi = fftw_plan_many_dft(rank, n, howmany, (fftw_complex *) P->Y, inembed, istride, idist, (fftw_complex *) P->Y, onembed, ostride, odist, sign, FT_FFTW_FLAGS);
    P->S = S;
    return P;
}

ft_spinsphere_fftw_plan * ft_plan_spinsph_synthesis(const int N, const int M, const int S) {
    const fftw_r2r_kind evenkind[2][1] = {{FFTW_REDFT01}, {FFTW_RODFT01}};
    const fftw_r2r_kind  oddkind[2][1] = {{FFTW_RODFT01}, {FFTW_REDFT01}};
    return ft_plan_spinsph_with_kind(N, M, S, S%2 == 0 ? evenkind : oddkind, FFTW_BACKWARD);
}

ft_spinsphere_fftw_plan * ft_plan_spinsph_analysis(const int N, const int M, const int S) {
    const fftw_r2r_kind evenkind[2][1] = {{FFTW_REDFT10}, {FFTW_RODFT10}};
    const fftw_r2r_kind  oddkind[2][1] = {{FFTW_RODFT10}, {FFTW_REDFT10}};
    return ft_plan_spinsph_with_kind(N, M, S, S%2 == 0 ? evenkind : oddkind, FFTW_FORWARD);
}

void ft_execute_spinsph_synthesis(const ft_spinsphere_fftw_plan * P, ft_complex * X, const int N, const int M) {
    if (P->S%2 == 0) {
        X[0][0] *= 2.0;
        X[0][1] *= 2.0;
        for (int j = 3; j < M; j += 4) {
            X[j*N][0] *= 2.0;
            X[j*N][1] *= 2.0;
            X[(j+1)*N][0] *= 2.0;
            X[(j+1)*N][1] *= 2.0;
        }
    }
    else {
        for (int j = 1; j < M-2; j += 4) {
            X[j*N][0] *= 2.0;
            X[j*N][1] *= 2.0;
            X[(j+1)*N][0] *= 2.0;
            X[(j+1)*N][1] *= 2.0;
        }
    }
    double * XD = (double *) X;
    fftw_execute_r2r(P->plantheta1, XD, XD);
    fftw_execute_r2r(P->plantheta1, XD+1, XD+1);
    fftw_execute_r2r(P->plantheta2, XD+2*N, XD+2*N);
    fftw_execute_r2r(P->plantheta2, XD+2*N+1, XD+2*N+1);
    fftw_execute_r2r(P->plantheta3, XD+4*N, XD+4*N);
    fftw_execute_r2r(P->plantheta3, XD+4*N+1, XD+4*N+1);
    fftw_execute_r2r(P->plantheta4, XD+6*N, XD+6*N);
    fftw_execute_r2r(P->plantheta4, XD+6*N+1, XD+6*N+1);
    for (int i = 0; i < 2*N*M; i++)
        XD[i] *= M_1_2_SQRT_2PI;
    colswap((const ft_complex *) X, (ft_complex *) P->Y, N, M);
    fftw_execute_dft(P->planphi, (fftw_complex *) P->Y, (fftw_complex *) X);
}

void ft_execute_spinsph_analysis(const ft_spinsphere_fftw_plan * P, ft_complex * X, const int N, const int M) {
    fftw_execute_dft(P->planphi, (fftw_complex *) X, (fftw_complex *) P->Y);
    colswap_t(X, (const ft_complex *) P->Y, N, M);
    double * XD = (double *) X;
    for (int i = 0; i < 2*N*M; i++)
        XD[i] *= M_2_SQRT_2PI/(2*N*M);
    fftw_execute_r2r(P->plantheta1, XD, XD);
    fftw_execute_r2r(P->plantheta1, XD+1, XD+1);
    fftw_execute_r2r(P->plantheta2, XD+2*N, XD+2*N);
    fftw_execute_r2r(P->plantheta2, XD+2*N+1, XD+2*N+1);
    fftw_execute_r2r(P->plantheta3, XD+4*N, XD+4*N);
    fftw_execute_r2r(P->plantheta3, XD+4*N+1, XD+4*N+1);
    fftw_execute_r2r(P->plantheta4, XD+6*N, XD+6*N);
    fftw_execute_r2r(P->plantheta4, XD+6*N+1, XD+6*N+1);
    if (P->S%2 == 0) {
        X[0][0] *= 0.5;
        X[0][1] *= 0.5;
        for (int j = 3; j < M; j += 4) {
            X[j*N][0] *= 0.5;
            X[j*N][1] *= 0.5;
            X[(j+1)*N][0] *= 0.5;
            X[(j+1)*N][1] *= 0.5;
        }
    }
    else {
        for (int j = 1; j < M-2; j += 4) {
            X[j*N][0] *= 0.5;
            X[j*N][1] *= 0.5;
            X[(j+1)*N][0] *= 0.5;
            X[(j+1)*N][1] *= 0.5;
        }
    }
}
