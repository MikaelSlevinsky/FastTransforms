#include "fasttransforms.h"
#include "ftutilities.h"

int main(int argc, const char * argv[]) {
    int checksum = 0;
    double err = 0.0;
    struct timeval start, end;

    static double * U1, * U2, * Us, * Ut;
    static double * V1, * V2, * V3, * V4, * V5, * V6;
    ft_gradient_plan * P1;
    ft_helmholtzhodge_plan * P2;

    int IERR, ITIME, N, M, NTIMES;

    if (argc > 1) {
        sscanf(argv[1], "%d", &IERR);
        if (argc > 2)
            sscanf(argv[2], "%d", &ITIME);
        else ITIME = 1;
    }
    else IERR = 1;

    printf("\nTesting the accuracy of the Helmholtz-Hodge decomposition.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;

        U1 = sphrand(N, M);
        U2 = sphrand(N, M);
        U1[0] = 0;
        U2[0] = 0;

        Us = calloc(N*M, sizeof(double));
        Ut = calloc(N*M, sizeof(double));

        V1 = calloc(N*M, sizeof(double));
        V2 = calloc(N*M, sizeof(double));
        V3 = calloc(N*M, sizeof(double));
        V4 = calloc(N*M, sizeof(double));
        V5 = malloc(N*M*sizeof(double));
        V6 = malloc(N*M*sizeof(double));

        P1 = ft_plan_sph_gradient(N);
        P2 = ft_plan_sph_helmholtzhodge(N);

        ft_execute_sph_gradient(P1, U1, V1, V2, N, M);
        ft_execute_sph_curl(P1, U2, V3, V4, N, M);
        for (int i = 0; i < N*M; i++) {
            V5[i] = V1[i] + V3[i];
            V6[i] = V2[i] + V4[i];
        }
        ft_execute_sph_helmholtzhodge(P2, Us, Ut, V5, V6, N, M);

        err = ft_norm_2arg(Us, U1, N*M)/ft_norm_1arg(U1, N*M);
        printf("ϵ_2 spheroidal \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, sqrt(N), &checksum);

        err = ft_normInf_2arg(Us, U1, N*M)/ft_normInf_1arg(U1, N*M);
        printf("ϵ_∞ spheroidal \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N, &checksum);

        err = ft_norm_2arg(Ut, U2, N*M)/ft_norm_1arg(U2, N*M);
        printf("ϵ_2 toroidal \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, sqrt(N), &checksum);

        err = ft_normInf_2arg(Ut, U2, N*M)/ft_normInf_1arg(U2, N*M);
        printf("ϵ_∞ toroidal \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N, &checksum);

        ft_destroy_gradient_plan(P1);
        ft_destroy_helmholtzhodge_plan(P2);
        free(U1);
        free(U2);
        free(Us);
        free(Ut);
        free(V1);
        free(V2);
        free(V3);
        free(V4);
        free(V5);
        free(V6);
    }

    printf("\nTiming the Helmholtz-Hodge decomposition.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        U1 = sphrand(N, M);
        U2 = sphrand(N, M);
        U1[0] = 0;
        U2[0] = 0;

        Us = calloc(N*M, sizeof(double));
        Ut = calloc(N*M, sizeof(double));

        V1 = calloc(N*M, sizeof(double));
        V2 = calloc(N*M, sizeof(double));
        V3 = calloc(N*M, sizeof(double));
        V4 = calloc(N*M, sizeof(double));
        V5 = malloc(N*M*sizeof(double));
        V6 = malloc(N*M*sizeof(double));

        P1 = ft_plan_sph_gradient(N);
        P2 = ft_plan_sph_helmholtzhodge(N);

        ft_execute_sph_gradient(P1, U1, V1, V2, N, M);
        ft_execute_sph_curl(P1, U2, V3, V4, N, M);
        for (int i = 0; i < N*M; i++) {
            V5[i] = V1[i] + V3[i];
            V6[i] = V2[i] + V4[i];
        }

        FT_TIME(ft_execute_sph_helmholtzhodge(P2, Us, Ut, V5, V6, N, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        printf("\n");
        ft_destroy_gradient_plan(P1);
        ft_destroy_helmholtzhodge_plan(P2);
        free(U1);
        free(U2);
        free(Us);
        free(Ut);
        free(V1);
        free(V2);
        free(V3);
        free(V4);
        free(V5);
        free(V6);
    }
    printf("];\n");

    return checksum;
}
