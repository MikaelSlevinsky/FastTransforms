#include "fasttransforms.h"
#include "utilities.h"

int main(int argc, const char * argv[]) {
    struct timeval start, end;

    fftw_init_threads();
    fftw_plan_with_nthreads(FT_GET_NUM_THREADS());

    static double * A;
    static double * B;
    SphericalHarmonicPlan * P;
    SphereFFTWPlan * PS, * PA;

    int IERR, ITIME, N, M, NLOOPS;

    if (argc > 1) {
        sscanf(argv[1], "%d", &IERR);
        if (argc > 2) sscanf(argv[2], "%d", &ITIME);
        else ITIME = 1;
    }
    else IERR = 1;

    printf("\nTesting the accuracy of spherical harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("err1 = [\n");
    for (int i = 0; i < IERR; i++) {
        N = 64*pow(2, i);
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
        freeSphericalHarmonicPlan(P);
        freeSphereFFTWPlan(PS);
        freeSphereFFTWPlan(PA);
    }
    printf("];\n");

    printf("\nTiming spherical harmonic transforms + FFTW synthesis and analysis.\n\n");
    printf("t1 = [\n");
    for (int i = 0; i < ITIME; i++) {
        N = 64*pow(2, i);
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
        freeSphericalHarmonicPlan(P);
        freeSphereFFTWPlan(PS);
        freeSphereFFTWPlan(PA);
    }
    printf("];\n");

    return 0;
}
