#include "fasttransforms.h"
#include "ftutilities.h"

int main(int argc, const char * argv[]) {
    int checksum = 0;
    double err = 0.0;
    struct timeval start, end;

    static double * A;
    static double * B;
    ft_harmonic_plan * P;
    ft_spin_harmonic_plan * SP;
    ft_sphere_fftw_plan * PS, * PA;
    ft_triangle_fftw_plan * QS, * QA;
    ft_disk_fftw_plan * RS, * RA;
    ft_rectdisk_fftw_plan * SS, * SA;
    ft_tetrahedron_fftw_plan * TS, * TA;
    ft_spinsphere_fftw_plan * US, * UA;
    //double alpha = -0.5, beta = -0.5, gamma = -0.5, delta = -0.5; // best case scenario
    double alpha = 0.0, beta = 0.0, gamma = 0.0, delta = 0.0; // not as good. perhaps better to transform to second kind Chebyshev

    int IERR, ITIME, J, N, L, M, NTIMES;


    if (argc > 1) {
        sscanf(argv[1], "%d", &IERR);
        if (argc > 2) {
            sscanf(argv[2], "%d", &ITIME);
            if (argc > 3) sscanf(argv[3], "%d", &J);
            else J = 0;
        }
        else ITIME = 1;
    }
    else IERR = 1;

    ft_fftw_init_threads();
    ft_fftw_plan_with_nthreads(FT_GET_NUM_THREADS());

    printf("\nTesting the accuracy of SHTs + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sph_synthesis(N, M);
        PA = ft_plan_sph_analysis(N, M);

        ft_execute_sph2fourier('N', P, A, N, M);
        ft_execute_sph_synthesis('N', PS, A, N, M);
        ft_execute_sph_analysis('N', PA, A, N, M);
        ft_execute_fourier2sph('N', P, A, N, M);

        ft_execute_sph2fourier('T', P, A, N, M);
        ft_execute_sph_synthesis('T', PA, A, N, M);
        ft_execute_sph_analysis('T', PS, A, N, M);
        ft_execute_fourier2sph('T', P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }

    printf("\nTiming spherical harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sph_synthesis(N, M);
        PA = ft_plan_sph_analysis(N, M);

        FT_TIME({ft_execute_sph2fourier('N', P, A, N, M); ft_execute_sph_synthesis('N', PS, A, N, M);}, start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME({ft_execute_sph_analysis('N', PA, A, N, M); ft_execute_fourier2sph('N', P, A, N, M);}, start, end, NTIMES)
        printf("  %.6f\n", elapsed(&start, &end, NTIMES));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of vector SHTs + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sphv_synthesis(N, M);
        PA = ft_plan_sphv_analysis(N, M);

        ft_execute_sphv2fourier('N', P, A, N, M);
        ft_execute_sphv_synthesis('N', PS, A, N, M);
        ft_execute_sphv_analysis('N', PA, A, N, M);
        ft_execute_fourier2sphv('N', P, A, N, M);

        ft_execute_sphv2fourier('T', P, A, N, M);
        ft_execute_sphv_synthesis('T', PA, A, N, M);
        ft_execute_sphv_analysis('T', PS, A, N, M);
        ft_execute_fourier2sphv('T', P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 2*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }

    printf("\nTiming spherical vector field transforms + FFTW synthesis and analysis.\n\n");
    printf("t2 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sphv_synthesis(N, M);
        PA = ft_plan_sphv_analysis(N, M);

        FT_TIME({ft_execute_sphv2fourier('N', P, A, N, M); ft_execute_sphv_synthesis('N', PS, A, N, M);}, start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME({ft_execute_sphv_analysis('N', PA, A, N, M); ft_execute_fourier2sphv('N', P, A, N, M);}, start, end, NTIMES)
        printf("  %.6f\n", elapsed(&start, &end, NTIMES));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Proriol² transforms + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = trirand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);
        QS = ft_plan_tri_synthesis(N, M);
        QA = ft_plan_tri_analysis(N, M);

        ft_execute_tri2cheb('N', P, A, N, M);
        ft_execute_tri_synthesis(QS, A, N, M);
        ft_execute_tri_analysis(QA, A, N, M);
        ft_execute_cheb2tri('N', P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 8*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_triangle_fftw_plan(QS);
        ft_destroy_triangle_fftw_plan(QA);
    }

    printf("\nTiming Proriol² transforms + FFTW synthesis and analysis.\n\n");
    printf("t3 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NTIMES = 1 + pow(2048/N, 2);

        A = trirand(N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);
        QS = ft_plan_tri_synthesis(N, M);
        QA = ft_plan_tri_analysis(N, M);

        FT_TIME({ft_execute_tri2cheb('N', P, A, N, M); ft_execute_tri_synthesis(QS, A, N, M);}, start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME({ft_execute_tri_analysis(QA, A, N, M); ft_execute_cheb2tri('N', P, A, N, M);}, start, end, NTIMES)
        printf("  %.6f\n", elapsed(&start, &end, NTIMES));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_triangle_fftw_plan(QS);
        ft_destroy_triangle_fftw_plan(QA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Zernike transforms + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_disk2cxf(N, alpha, beta);
        RS = ft_plan_disk_synthesis(N, M);
        RA = ft_plan_disk_analysis(N, M);

        ft_execute_disk2cxf('N', P, A, N, M);
        ft_execute_disk_synthesis(RS, A, N, M);
        ft_execute_disk_analysis(RA, A, N, M);
        ft_execute_cxf2disk('N', P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 8*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_disk_fftw_plan(RS);
        ft_destroy_disk_fftw_plan(RA);
    }

    printf("\nTiming Zernike transforms + FFTW synthesis and analysis.\n\n");
    printf("t4 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NTIMES = 1 + pow(2048/N, 2);

        A = diskrand(N, M);
        P = ft_plan_disk2cxf(N, alpha, beta);
        RS = ft_plan_disk_synthesis(N, M);
        RA = ft_plan_disk_analysis(N, M);

        FT_TIME({ft_execute_disk2cxf('N', P, A, N, M); ft_execute_disk_synthesis(RS, A, N, M);}, start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME({ft_execute_disk_analysis(RA, A, N, M); ft_execute_cxf2disk('N', P, A, N, M);}, start, end, NTIMES)
        printf("  %.6f\n", elapsed(&start, &end, NTIMES));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_disk_fftw_plan(RS);
        ft_destroy_disk_fftw_plan(RA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Dunkl-Xu transforms + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = rectdiskrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_rectdisk2cheb(N, beta);
        SS = ft_plan_rectdisk_synthesis(N, M);
        SA = ft_plan_rectdisk_analysis(N, M);

        ft_execute_rectdisk2cheb('N', P, A, N, M);
        ft_execute_rectdisk_synthesis(SS, A, N, M);
        ft_execute_rectdisk_analysis(SA, A, N, M);
        ft_execute_cheb2rectdisk('N', P, A, N, M);

        err = ft_norm_2arg(A, B, N*M)/ft_norm_1arg(B, N*M);
        printf("ϵ_2 \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 8*sqrt(N), &checksum);
        err = ft_normInf_2arg(A, B, N*M)/ft_normInf_1arg(B, N*M);
        printf("ϵ_∞ \t\t\t (N×M) = (%5ix%5i): \t |%20.2e ", N, M, err);
        ft_checktest(err, 4*N, &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_rectdisk_fftw_plan(SS);
        ft_destroy_rectdisk_fftw_plan(SA);
    }

    printf("\nTiming Dunkl-Xu transforms + FFTW synthesis and analysis.\n\n");
    printf("t5 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NTIMES = 1 + pow(2048/N, 2);

        A = rectdiskrand(N, M);
        P = ft_plan_rectdisk2cheb(N, beta);
        SS = ft_plan_rectdisk_synthesis(N, M);
        SA = ft_plan_rectdisk_analysis(N, M);

        FT_TIME({ft_execute_rectdisk2cheb('N', P, A, N, M); ft_execute_rectdisk_synthesis(SS, A, N, M);}, start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME({ft_execute_rectdisk_analysis(SA, A, N, M); ft_execute_cheb2rectdisk('N', P, A, N, M);}, start, end, NTIMES)
        printf("  %.6f\n", elapsed(&start, &end, NTIMES));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_rectdisk_fftw_plan(SS);
        ft_destroy_rectdisk_fftw_plan(SA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of Proriol³ transforms + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;

        A = tetrand(N, L, M);
        B = copymat(A, N, L*M);
        P = ft_plan_tet2cheb(N, alpha, beta, gamma, delta);
        TS = ft_plan_tet_synthesis(N, L, M);
        TA = ft_plan_tet_analysis(N, L, M);

        ft_execute_tet2cheb(P, A, N, L, M);
        ft_execute_tet_synthesis(TS, A, N, L, M);
        ft_execute_tet_analysis(TA, A, N, L, M);
        ft_execute_cheb2tet(P, A, N, L, M);

        err = ft_norm_2arg(A, B, N*L*M)/ft_norm_1arg(B, N*L*M);
        printf("ϵ_2 \t\t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 8*pow(N*L*M, 1.0/3.0), &checksum);
        err = ft_normInf_2arg(A, B, N*L*M)/ft_normInf_1arg(B, N*L*M);
        printf("ϵ_∞ \t\t (N×L×M) = (%5ix%5i×%5i): \t |%20.2e ", N, L, M, err);
        ft_checktest(err, 4*pow(N*L*M, 2.0/3.0), &checksum);

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_tetrahedron_fftw_plan(TS);
        ft_destroy_tetrahedron_fftw_plan(TA);
    }

    printf("\nTiming Proriol³ transforms + FFTW synthesis and analysis.\n\n");
    printf("t6 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 16*pow(2, i)+J;
        L = M = N;
        NTIMES = 1 + pow(256/N, 2);

        A = tetrand(N, L, M);
        P = ft_plan_tet2cheb(N, alpha, beta, gamma, delta);
        TS = ft_plan_tet_synthesis(N, L, M);
        TA = ft_plan_tet_analysis(N, L, M);

        FT_TIME({ft_execute_tet2cheb(P, A, N, L, M); ft_execute_tet_synthesis(TS, A, N, L, M);}, start, end, NTIMES)
        printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

        FT_TIME({ft_execute_tet_analysis(TA, A, N, L, M); ft_execute_cheb2tet(P, A, N, L, M);}, start, end, NTIMES)
        printf("  %.6f\n", elapsed(&start, &end, NTIMES));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_tetrahedron_fftw_plan(TS);
        ft_destroy_tetrahedron_fftw_plan(TA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spin-weighted SHTs + FFTW synthesis and analysis.\n\n");
    printf("\t\t\t Test \t\t\t\t |        Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        for (int S = -4; S <= 4; S++) {
            ft_complex * AC = spinsphrand(N, M, S);
            ft_complex * BC = (ft_complex *) copymat((double *) AC, 2*N, M);
            A = (double *) AC;
            B = (double *) BC;

            SP = ft_plan_spinsph2fourier(N, S);
            US = ft_plan_spinsph_synthesis(N, M, S);
            UA = ft_plan_spinsph_analysis(N, M, S);

            ft_execute_spinsph2fourier('N', SP, AC, N, M);
            ft_execute_spinsph_synthesis(US, AC, N, M);
            ft_execute_spinsph_analysis(UA, AC, N, M);
            ft_execute_fourier2spinsph('N', SP, AC, N, M);

            err = ft_norm_2arg(A, B, 2*N*M)/ft_norm_1arg(B, 2*N*M);
            printf("ϵ_2 \t\t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
            ft_checktest(err, 4*sqrt(N), &checksum);
            err = ft_normInf_2arg(A, B, 2*N*M)/ft_normInf_1arg(B, 2*N*M);
            printf("ϵ_∞ \t\t\t (N×M, S) = (%5ix%5i,%3i): \t |%20.2e ", N, M, S, err);
            ft_checktest(err, 2*N, &checksum);

            free(AC);
            free(BC);
            ft_destroy_spin_harmonic_plan(SP);
            ft_destroy_spinsphere_fftw_plan(US);
            ft_destroy_spinsphere_fftw_plan(UA);
        }
    }

    printf("\nTiming spin-weighted SHTs + FFTW synthesis and analysis.\n\n");
    printf("t7 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NTIMES = 1 + pow(2048/N, 2);

        for (int S = -2; S <= 2; S++) {
            ft_complex * AC = spinsphrand(N, M, S);
            SP = ft_plan_spinsph2fourier(N, S);
            US = ft_plan_spinsph_synthesis(N, M, S);
            UA = ft_plan_spinsph_analysis(N, M, S);

            FT_TIME({ft_execute_spinsph2fourier('N', SP, AC, N, M); ft_execute_spinsph_synthesis(US, AC, N, M);}, start, end, NTIMES)
            printf("%d  %.6f", N, elapsed(&start, &end, NTIMES));

            FT_TIME({ft_execute_spinsph_analysis(UA, AC, N, M); ft_execute_fourier2spinsph('N', SP, AC, N, M);}, start, end, NTIMES)
            printf("  %.6f\n", elapsed(&start, &end, NTIMES));

            free(AC);
            ft_destroy_spin_harmonic_plan(SP);
            ft_destroy_spinsphere_fftw_plan(US);
            ft_destroy_spinsphere_fftw_plan(UA);
        }
    }
    printf("];\n");

    return checksum;
}
