#include "fasttransforms.h"
#include "ftutilities.h"

int main(int argc, const char * argv[]) {
    struct timeval start, end;

    static double * A;
    static double * B;
    ft_harmonic_plan * P;
    ft_sphere_fftw_plan * PS, * PA;
    ft_triangle_fftw_plan * QS, * QA;
    ft_disk_fftw_plan * RS, * RA;
    //double alpha = -0.5, beta = -0.5, gamma = -0.5; // best case scenario
    double alpha = 0.0, beta = 0.0, gamma = 0.0; // not as good. perhaps better to transform to second kind Chebyshev

    int IERR, ITIME, J, N, M, NLOOPS;


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

    fftw_init_threads();
    fftw_plan_with_nthreads(FT_GET_NUM_THREADS());

    printf("\nTesting the accuracy of spherical harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("err1 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sph_synthesis(N, M);
        PA = ft_plan_sph_analysis(N, M);

        ft_execute_sph2fourier(P, A, N, M);
        ft_execute_sph_synthesis(PS, A, N, M);
        ft_execute_sph_analysis(PA, A, N, M);
        ft_execute_fourier2sph(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sph_synthesis(N, M);
        PA = ft_plan_sph_analysis(N, M);

        ft_execute_sph_synthesis(PS, A, N, M);
        ft_execute_sph_analysis(PA, A, N, M);
        ft_execute_sph_synthesis(PS, A, N, M);
        ft_execute_sph_analysis(PA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph2fourier(P, A, N, M);
            ft_execute_sph_synthesis(PS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sph_analysis(PA, A, N, M);
            ft_execute_fourier2sph(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of spherical vector field transforms + FFTW synthesis and analysis.\n\n");
    printf("err2 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;

        A = sphrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sphv_synthesis(N, M);
        PA = ft_plan_sphv_analysis(N, M);

        ft_execute_sphv2fourier(P, A, N, M);
        ft_execute_sphv_synthesis(PS, A, N, M);
        ft_execute_sphv_analysis(PA, A, N, M);
        ft_execute_fourier2sphv(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }
    printf("];\n");

    printf("\nTiming spherical vector field transforms + FFTW synthesis and analysis.\n\n");
    printf("t2 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = ft_plan_sph2fourier(N);
        PS = ft_plan_sphv_synthesis(N, M);
        PA = ft_plan_sphv_analysis(N, M);

        ft_execute_sphv_synthesis(PS, A, N, M);
        ft_execute_sphv_analysis(PA, A, N, M);
        ft_execute_sphv_synthesis(PS, A, N, M);
        ft_execute_sphv_analysis(PA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv2fourier(P, A, N, M);
            ft_execute_sphv_synthesis(PS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_sphv_analysis(PA, A, N, M);
            ft_execute_fourier2sphv(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_sphere_fftw_plan(PS);
        ft_destroy_sphere_fftw_plan(PA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of triangular harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("err3 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = trirand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);
        QS = ft_plan_tri_synthesis(N, M);
        QA = ft_plan_tri_analysis(N, M);

        ft_execute_tri2cheb(P, A, N, M);
        ft_execute_tri_synthesis(QS, A, N, M);
        ft_execute_tri_analysis(QA, A, N, M);
        ft_execute_cheb2tri(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_triangle_fftw_plan(QS);
        ft_destroy_triangle_fftw_plan(QA);
    }
    printf("];\n");

    printf("\nTiming triangular harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t3 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NLOOPS = 1 + pow(2048/N, 2);

        A = trirand(N, M);
        P = ft_plan_tri2cheb(N, alpha, beta, gamma);
        QS = ft_plan_tri_synthesis(N, M);
        QA = ft_plan_tri_analysis(N, M);

        ft_execute_tri_synthesis(QS, A, N, M);
        ft_execute_tri_analysis(QA, A, N, M);
        ft_execute_tri_synthesis(QS, A, N, M);
        ft_execute_tri_analysis(QA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri2cheb(P, A, N, M);
            ft_execute_tri_synthesis(QS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_tri_analysis(QA, A, N, M);
            ft_execute_cheb2tri(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_triangle_fftw_plan(QS);
        ft_destroy_triangle_fftw_plan(QA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of disk harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("err4 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskrand(N, M);
        B = copymat(A, N, M);
        P = ft_plan_disk2cxf(N);
        RS = ft_plan_disk_synthesis(N, M);
        RA = ft_plan_disk_analysis(N, M);

        ft_execute_disk2cxf(P, A, N, M);
        ft_execute_disk_synthesis(RS, A, N, M);
        ft_execute_disk_analysis(RA, A, N, M);
        ft_execute_cxf2disk(P, A, N, M);

        printf("%1.2e  ", norm_2arg(A, B, N*M)/norm_1arg(B, N*M));
        printf("%1.2e\n", normInf_2arg(A, B, N*M)/normInf_1arg(B, N*M));

        free(A);
        free(B);
        ft_destroy_harmonic_plan(P);
        ft_destroy_disk_fftw_plan(RS);
        ft_destroy_disk_fftw_plan(RA);
    }
    printf("];\n");

    printf("\nTiming disk harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t4 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NLOOPS = 1 + pow(2048/N, 2);

        A = diskrand(N, M);
        P = ft_plan_disk2cxf(N);
        RS = ft_plan_disk_synthesis(N, M);
        RA = ft_plan_disk_analysis(N, M);

        ft_execute_disk_synthesis(RS, A, N, M);
        ft_execute_disk_analysis(RA, A, N, M);
        ft_execute_disk_synthesis(RS, A, N, M);
        ft_execute_disk_analysis(RA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk2cxf(P, A, N, M);
            ft_execute_disk_synthesis(RS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            ft_execute_disk_analysis(RA, A, N, M);
            ft_execute_cxf2disk(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        ft_destroy_harmonic_plan(P);
        ft_destroy_disk_fftw_plan(RS);
        ft_destroy_disk_fftw_plan(RA);
    }
    printf("];\n");

    return 0;
}
