#include "fasttransforms.h"
#include "ftutilities.h"

// Q = [cos(α) -sin(α) 0; sin(α) cos(α) 0; 0 0 1]*[cos(β) 0 -sin(β); 0 1 0; sin(β) 0 cos(β)]*[cos(γ) -sin(γ) 0; sin(γ) cos(γ) 0; 0 0 1]

int main(int argc, const char * argv[]) {
    int checksum = 0;
    double err = 0.0;
    struct timeval start, end;

    static double * A;
    static double * B;
    ft_orthogonal_transformation Q, U;
    ft_reflection W;
    ft_ZYZR F;
    ft_sph_isometry_plan * J;
    double alpha, beta, gamma;

    int IERR, ITIME, N, L, M, NTIMES;

    if (argc > 1) {
        sscanf(argv[1], "%d", &IERR);
        if (argc > 2)
            sscanf(argv[2], "%d", &ITIME);
        else ITIME = 1;
    }
    else IERR = 1;

    printf("\nTesting the computation of structured isometries.\n\n");
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    Q = (ft_orthogonal_transformation) {0.902113004769273, 0.38751720202221734, 0.18979606097868743, -0.38355704238148136, 0.9216490856090721, -0.05871080169382652, -0.19767681165408388, -0.019833838076209875, 0.9800665778412416};
    W = (ft_reflection) {0.0, 0.0, 1.0};
    F = ft_create_ZYZR(Q);

    alpha = atan2(F.s[0], F.c[0]);
    beta = atan2(F.s[1], F.c[1]);
    gamma = atan2(F.s[2], F.c[2]);
    err = sqrt((alpha-0.1)*(alpha-0.1) + (beta-0.2)*(beta-0.2) + (gamma-0.3)*(gamma-0.3))/sqrt(0.1*0.1+0.2*0.2+0.3*0.3);

    printf("Euler angles from an orthogonal transformation: \t |%20.2e ", err);
    ft_checktest(err+(F.sign-1), 2, &checksum);

    U = (ft_orthogonal_transformation) {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    ft_apply_ZYZR(F, &U);

    err = ft_norm_2arg(Q.Q, U.Q, 9)/ft_norm_1arg(Q.Q, 9);

    printf("Reconstruction error from ZYZR, a rotation: \t\t |%20.2e ", err);
    ft_checktest(err, 2, &checksum);

    ft_apply_reflection(W, &Q);

    F = ft_create_ZYZR(Q);

    U = (ft_orthogonal_transformation) {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    ft_apply_ZYZR(F, &U);

    err = ft_norm_2arg(Q.Q, U.Q, 9)/ft_norm_1arg(Q.Q, 9);

    printf("Reconstruction error from ZYZR, a reflection: \t\t |%20.2e ", err);
    ft_checktest(err, 2, &checksum);

    U = (ft_orthogonal_transformation) {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    ft_apply_reflection(W, &U);
    F = ft_create_ZYZR(U);

    alpha = atan2(F.s[0], F.c[0]);
    beta = atan2(F.s[1], F.c[1]);
    gamma = atan2(F.s[2], F.c[2]);
    err = sqrt(alpha*alpha + beta*beta + gamma*gamma);

    printf("Euler angles from an xy-plane reflection: \t\t |%20.2e ", err);
    ft_checktest(err+(F.sign+1), 2, &checksum);


    printf("\nTesting the accuracy of spherical isometries.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        J = ft_plan_sph_isometry(N);
        alpha = 0.1;
        beta = 0.2;
        gamma = 0.3;
        W = (ft_reflection) {0.123, 0.456, 0.789}; // {0.0, 0.0, 1.0}; //

        ft_execute_sph_ZY_axis_exchange(J, A, N, M);
        ft_execute_sph_ZY_axis_exchange(J, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 J^2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ J^2 \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N*N, &checksum);

        ft_execute_sph_rotation(J, alpha, beta, gamma, A, N, M);
        ft_execute_sph_rotation(J, -gamma, -beta, -alpha, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 rotation \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ rotation \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N*N, &checksum);

        ft_execute_sph_reflection(J, W, A, N, M);
        ft_execute_sph_reflection(J, W, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 reflection \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ reflection \t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, N*N, &checksum);

        free(A);
        free(B);
        ft_destroy_sph_isometry_plan(J);
    }

    printf("\nTiming spherical isometries.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i);
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        J = ft_plan_sph_isometry(N);
        alpha = 0.1;
        beta = 0.2;
        gamma = 0.3;

        FT_TIME(ft_execute_sph_rotation(J, alpha, beta, gamma, A, N, M), start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        printf("\n");
        free(A);
        ft_destroy_sph_isometry_plan(J);
    }
    printf("];\n");

    return checksum;
}
