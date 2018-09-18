#include "fasttransforms.h"
#include "utilities.h"

int main(int argc, const char * argv[]) {
    struct timeval start, end;

    static double * A;
    static double * B;
    HarmonicPlan * P;
    SphereFFTWPlan * PS, * PA;
    TriangleFFTWPlan * QS, * QA;
    DiskFFTWPlan * RS, * RA;
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
        B = copyA(A, N, M);
        P = plan_sph2fourier(N);
        PS = plan_sph_synthesis(N, M);
        PA = plan_sph_analysis(N, M);

        execute_sph2fourier(P, A, N, M);
        execute_sph_synthesis(PS, A, N, M);
        execute_sph_analysis(PA, A, N, M);
        execute_fourier2sph(P, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        freeHarmonicPlan(P);
        freeSphereFFTWPlan(PS);
        freeSphereFFTWPlan(PA);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 2*N-1;
        NLOOPS = 1 + pow(2048/N, 2);

        A = sphrand(N, M);
        P = plan_sph2fourier(N);
        PS = plan_sph_synthesis(N, M);
        PA = plan_sph_analysis(N, M);

        execute_sph_synthesis(PS, A, N, M);
        execute_sph_analysis(PA, A, N, M);
        execute_sph_synthesis(PS, A, N, M);
        execute_sph_analysis(PA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph2fourier(P, A, N, M);
            execute_sph_synthesis(PS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_sph_analysis(PA, A, N, M);
            execute_fourier2sph(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        freeHarmonicPlan(P);
        freeSphereFFTWPlan(PS);
        freeSphereFFTWPlan(PA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of triangular harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("err2 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = N;

        A = trirand(N, M);
        B = copyA(A, N, M);
        P = plan_tri2cheb(N, alpha, beta, gamma);
        QS = plan_tri_synthesis(N, M);
        QA = plan_tri_analysis(N, M);

        execute_tri2cheb(P, A, N, M);
        execute_tri_synthesis(QS, A, N, M);
        execute_tri_analysis(QA, A, N, M);
        execute_cheb2tri(P, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        freeHarmonicPlan(P);
        freeTriangleFFTWPlan(QS);
        freeTriangleFFTWPlan(QA);
    }
    printf("];\n");

    printf("\nTiming triangular harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t2 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = N;
        NLOOPS = 1 + pow(2048/N, 2);

        A = trirand(N, M);
        P = plan_tri2cheb(N, alpha, beta, gamma);
        QS = plan_tri_synthesis(N, M);
        QA = plan_tri_analysis(N, M);

        execute_tri_synthesis(QS, A, N, M);
        execute_tri_analysis(QA, A, N, M);
        execute_tri_synthesis(QS, A, N, M);
        execute_tri_analysis(QA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri2cheb(P, A, N, M);
            execute_tri_synthesis(QS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_tri_analysis(QA, A, N, M);
            execute_cheb2tri(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        freeHarmonicPlan(P);
        freeTriangleFFTWPlan(QS);
        freeTriangleFFTWPlan(QA);
    }
    printf("];\n");

    printf("\nTesting the accuracy of disk harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("err3 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;

        A = diskrand(N, M);
        B = copyA(A, N, M);
        P = plan_disk2cxf(N);
        RS = plan_disk_synthesis(N, M);
        RA = plan_disk_analysis(N, M);

        execute_disk2cxf(P, A, N, M);
        execute_disk_synthesis(RS, A, N, M);
        execute_disk_analysis(RA, A, N, M);
        execute_cxf2disk(P, A, N, M);

        printf("%1.2e  ", vecnorm_2arg(A, B, N, M)/vecnorm_1arg(B, N, M));
        printf("%1.2e\n", vecnormInf_2arg(A, B, N, M)/vecnormInf_1arg(B, N, M));

        free(A);
        free(B);
        freeHarmonicPlan(P);
        freeDiskFFTWPlan(RS);
        freeDiskFFTWPlan(RA);
    }
    printf("];\n");

    printf("\nTiming disk harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t3 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i)+J;
        M = 4*N-3;
        NLOOPS = 1 + pow(2048/N, 2);

        A = diskrand(N, M);
        P = plan_disk2cxf(N);
        RS = plan_disk_synthesis(N, M);
        RA = plan_disk_analysis(N, M);

        execute_disk_synthesis(RS, A, N, M);
        execute_disk_analysis(RA, A, N, M);
        execute_disk_synthesis(RS, A, N, M);
        execute_disk_analysis(RA, A, N, M);

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk2cxf(P, A, N, M);
            execute_disk_synthesis(RS, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("%d  %.6f", N, elapsed(&start, &end, NLOOPS));

        gettimeofday(&start, NULL);
        for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
            execute_disk_analysis(RA, A, N, M);
            execute_cxf2disk(P, A, N, M);
        }
        gettimeofday(&end, NULL);

        printf("  %.6f\n", elapsed(&start, &end, NLOOPS));

        free(A);
        freeHarmonicPlan(P);
        freeDiskFFTWPlan(RS);
        freeDiskFFTWPlan(RA);
    }
    printf("];\n");

    return 0;
}
