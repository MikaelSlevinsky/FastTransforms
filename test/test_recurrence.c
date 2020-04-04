#include "fasttransforms.h"
#include "ftinternal.h"
#include "ftutilities.h"

int main(void) {
    int checksum = 0;
    int NLOOPS = 250;
    struct timeval start, end;

    int MN = 1024;

    unsigned int eax = 1, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    ft_simd simd = (ft_simd) {edx & bit_SSE, edx & bit_SSE2, ecx & bit_AVX, ecx & bit_FMA, ebx & bit_AVX512F};
    simd.sse ? printf("SSE detected.\t\t"GREEN("✓")"\n") : printf("SSE not detected.\t"RED("×")"\n");
    simd.sse2 ? printf("SSE2 detected.\t\t"GREEN("✓")"\n") : printf("SSE2 not detected.\t"RED("×")"\n");
    simd.avx ? printf("AVX detected.\t\t"GREEN("✓")"\n") : printf("AVX not detected.\t"RED("×")"\n");
    simd.fma ? printf("FMA detected.\t\t"GREEN("✓")"\n") : printf("FMA not detected.\t"RED("×")"\n");
    simd.avx512f ? printf("AVX512F detected.\t"GREEN("✓")"\n") : printf("AVX512F not detected.\t"RED("×")"\n");

    printf("\nTesting methods for Horner summation.\n");
    printf("\n\tSingle precision.\n\n");

    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 256; n < 257; n++) {
        for (int m = 256; m < 264; m++) {
            float * c = malloc(n*sizeof(float));
            float * x = malloc(m*sizeof(float));
            float * f = malloc(m*sizeof(float));
            float * fd = malloc(m*sizeof(float));
            float err = 0.0f;
            for (int k = 0; k < n; k++)
                c[k] = 1.0f/(k+1.0f);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0f+2.0f*j/(m-1.0f));

            horner_defaultf(n, c, 1, m, x, fd);
            horner_SSEf(n, c, 1, m, x, f);
            err = powf(ft_norm_2argf(f, fd, m), 2);
            horner_AVXf(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            horner_AVX_FMAf(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            horner_AVX512Ff(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            ft_hornerf(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            err = sqrtf(err);
            err /= ft_norm_1argf(fd, m);

            printf("(m×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", m, n, (double) err);
            ft_checktestf(err, n, &checksum);
            free(c);
            free(x);
            free(f);
            free(fd);
        }
    }

    for (int n = MN; n < 2*MN; n *= 2) {
        for (int m = MN; m < 2*MN; m *= 2) {
            float * c = malloc(n*sizeof(float));
            float * x = malloc(m*sizeof(float));
            float * f = malloc(m*sizeof(float));
            for (int k = 0; k < n; k++)
                c[k] = 1.0f/(k+1.0f);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0f+2.0f*j/(m-1.0f));

            horner_defaultf(n, c, 1, m, x, f);
            horner_defaultf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_defaultf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_defaultf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            horner_SSEf(n, c, 1, m, x, f);
            horner_SSEf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_SSEf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_SSEf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            horner_AVXf(n, c, 1, m, x, f);
            horner_AVXf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_AVXf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_AVXf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            horner_AVX_FMAf(n, c, 1, m, x, f);
            horner_AVX_FMAf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_AVX_FMAf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_AVX_FMAf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            horner_AVX512Ff(n, c, 1, m, x, f);
            horner_AVX512Ff(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_AVX512Ff(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_AVX512Ff \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            ft_hornerf(n, c, 1, m, x, f);
            ft_hornerf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                ft_hornerf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for ft_hornerf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            free(c);
            free(x);
            free(f);
        }
    }

    printf("\n\tDouble precision.\n\n");

    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 256; n < 257; n++) {
        for (int m = 256; m < 264; m++) {
            double * c = malloc(n*sizeof(double));
            double * x = malloc(m*sizeof(double));
            double * f = malloc(m*sizeof(double));
            double * fd = malloc(m*sizeof(double));
            double err = 0.0;
            for (int k = 0; k < n; k++)
                c[k] = 1.0/(k+1.0);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0+2.0*j/(m-1.0));

            horner_default(n, c, 1, m, x, fd);
            horner_SSE2(n, c, 1, m, x, f);
            err = pow(ft_norm_2arg(f, fd, m), 2);
            horner_AVX(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            horner_AVX_FMA(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            horner_AVX512F(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            ft_horner(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            err = sqrt(err);
            err /= ft_norm_1arg(fd, m);

            printf("(m×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", m, n, (double) err);
            ft_checktest(err, n, &checksum);
            free(c);
            free(x);
            free(f);
            free(fd);
        }
    }

    for (int n = MN; n < 2*MN; n *= 2) {
        for (int m = MN; m < 2*MN; m *= 2) {
            double * c = malloc(n*sizeof(double));
            double * x = malloc(m*sizeof(double));
            double * f = malloc(m*sizeof(double));
            for (int k = 0; k < n; k++)
                c[k] = 1.0/(k+1.0);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0+2.0*j/(m-1.0));

            horner_default(n, c, 1, m, x, f);
            horner_default(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_default(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_default \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            horner_SSE2(n, c, 1, m, x, f);
            horner_SSE2(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_SSE2(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_SSE2 \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            horner_AVX(n, c, 1, m, x, f);
            horner_AVX(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_AVX(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_AVX \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            horner_AVX_FMA(n, c, 1, m, x, f);
            horner_AVX_FMA(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_AVX_FMA(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_AVX_FMA \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            horner_AVX512F(n, c, 1, m, x, f);
            horner_AVX512F(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                horner_AVX512F(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for horner_AVX512F \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            ft_horner(n, c, 1, m, x, f);
            ft_horner(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                ft_horner(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for ft_horner \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            free(c);
            free(x);
            free(f);
        }
    }

    printf("\nTesting methods for Clenshaw summation.\n");
    printf("\n\tSingle precision.\n\n");

    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 256; n < 257; n++) {
        for (int m = 256; m < 264; m++) {
            float * c = malloc(n*sizeof(float));
            float * x = malloc(m*sizeof(float));
            float * f = malloc(m*sizeof(float));
            float * fd = malloc(m*sizeof(float));
            float err = 0.0f;
            for (int k = 0; k < n; k++)
                c[k] = 1.0f/(k+1.0f);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0f+2.0f*j/(m-1.0f));

            clenshaw_defaultf(n, c, 1, m, x, fd);
            clenshaw_SSEf(n, c, 1, m, x, f);
            err = powf(ft_norm_2argf(f, fd, m), 2);
            clenshaw_AVXf(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            clenshaw_AVX_FMAf(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            clenshaw_AVX512Ff(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            ft_clenshawf(n, c, 1, m, x, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            err = sqrtf(err);
            err /= ft_norm_1argf(fd, m);

            printf("(m×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", m, n, (double) err);
            ft_checktestf(err, n, &checksum);
            free(c);
            free(x);
            free(f);
            free(fd);
        }
    }

    for (int n = MN; n < 2*MN; n *= 2) {
        for (int m = MN; m < 2*MN; m *= 2) {
            float * c = malloc(n*sizeof(float));
            float * x = malloc(m*sizeof(float));
            float * f = malloc(m*sizeof(float));
            for (int k = 0; k < n; k++)
                c[k] = 1.0f/(k+1.0f);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0f+2.0f*j/(m-1.0f));

            clenshaw_defaultf(n, c, 1, m, x, f);
            clenshaw_defaultf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_defaultf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_defaultf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            clenshaw_SSEf(n, c, 1, m, x, f);
            clenshaw_SSEf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_SSEf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_SSEf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            clenshaw_AVXf(n, c, 1, m, x, f);
            clenshaw_AVXf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_AVXf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_AVXf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            clenshaw_AVX_FMAf(n, c, 1, m, x, f);
            clenshaw_AVX_FMAf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_AVX_FMAf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_AVX_FMAf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            clenshaw_AVX512Ff(n, c, 1, m, x, f);
            clenshaw_AVX512Ff(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_AVX512Ff(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_AVX512Ff \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            ft_clenshawf(n, c, 1, m, x, f);
            ft_clenshawf(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                ft_clenshawf(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for ft_clenshawf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            free(c);
            free(x);
            free(f);
        }
    }

    printf("\n\tDouble precision.\n\n");

    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 256; n < 257; n++) {
        for (int m = 256; m < 264; m++) {
            double * c = malloc(n*sizeof(double));
            double * x = malloc(m*sizeof(double));
            double * f = malloc(m*sizeof(double));
            double * fd = malloc(m*sizeof(double));
            double err = 0.0;
            for (int k = 0; k < n; k++)
                c[k] = 1.0/(k+1.0);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0+2.0*j/(m-1.0));

            clenshaw_default(n, c, 1, m, x, fd);
            clenshaw_SSE2(n, c, 1, m, x, f);
            err = pow(ft_norm_2arg(f, fd, m), 2);
            clenshaw_AVX(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            clenshaw_AVX_FMA(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            clenshaw_AVX512F(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            ft_clenshaw(n, c, 1, m, x, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            err = sqrt(err);
            err /= ft_norm_1arg(fd, m);

            printf("(m×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", m, n, (double) err);
            ft_checktest(err, n, &checksum);
            free(c);
            free(x);
            free(f);
            free(fd);
        }
    }

    for (int n = MN; n < 2*MN; n *= 2) {
        for (int m = MN; m < 2*MN; m *= 2) {
            double * c = malloc(n*sizeof(double));
            double * x = malloc(m*sizeof(double));
            double * f = malloc(m*sizeof(double));
            for (int k = 0; k < n; k++)
                c[k] = 1.0/(k+1.0);
            for (int j = 0; j < m; j++)
                x[j] = (-1.0+2.0*j/(m-1.0));

            clenshaw_default(n, c, 1, m, x, f);
            clenshaw_default(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_default(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_default \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            clenshaw_SSE2(n, c, 1, m, x, f);
            clenshaw_SSE2(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_SSE2(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_SSE2 \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            clenshaw_AVX(n, c, 1, m, x, f);
            clenshaw_AVX(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_AVX(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_AVX \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            clenshaw_AVX_FMA(n, c, 1, m, x, f);
            clenshaw_AVX_FMA(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_AVX_FMA(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_AVX_FMA \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            clenshaw_AVX512F(n, c, 1, m, x, f);
            clenshaw_AVX512F(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                clenshaw_AVX512F(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for clenshaw_AVX512F \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            ft_clenshaw(n, c, 1, m, x, f);
            ft_clenshaw(n, c, 1, m, x, f);
            gettimeofday(&start, NULL);
            for (int ntimes = 0; ntimes < NLOOPS; ntimes++)
                ft_clenshaw(n, c, 1, m, x, f);
            gettimeofday(&end, NULL);
            printf("Time for ft_clenshaw \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

            free(c);
            free(x);
            free(f);
        }
    }
    printf("\n");
    return checksum;
}
