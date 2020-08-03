#include "fasttransforms.h"
#include "ftutilities.h"

int main(void) {
    int checksum = 0;
    int NTIMES = 100;
    struct timeval start, end;

    int MN = 1024;

    ft_simd simd = get_simd();
    simd.sse ? printf("SSE detected.\t\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("SSE not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    simd.sse2 ? printf("SSE2 detected.\t\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("SSE2 not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    simd.avx ? printf("AVX detected.\t\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("AVX not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    simd.avx2 ? printf("AVX2 detected.\t\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("AVX2 not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    simd.fma ? printf("FMA detected.\t\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("FMA not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    simd.avx512f ? printf("AVX512F detected.\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("AVX512F not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    simd.neon ? printf("NEON detected.\t\t\t\t\t\t\t\t       "GREEN("✓")"\n") : printf("NEON not detected.\t\t\t\t\t\t\t       "RED("✗")"\n");
    printf("The "CYAN("sizeof(ft_simd)")" is \t\t\t\t\t\t\t       %li\n", sizeof(ft_simd));

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
            #if defined(__i386__) || defined(__x86_64__)
                horner_SSEf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                horner_AVXf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                horner_AVX_FMAf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                horner_AVX512Ff(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
            #elif defined(__aarch64__)
                horner_NEONf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
            #endif
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

            FT_TIME(horner_defaultf(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for horner_defaultf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(horner_SSEf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_SSEf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(horner_AVXf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_AVXf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(horner_AVX_FMAf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_AVX_FMAf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(horner_AVX512Ff(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_AVX512Ff \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(horner_NEONf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_NEONf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #endif

            FT_TIME(ft_hornerf(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for ft_hornerf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

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
            #if defined(__i386__) || defined(__x86_64__)
                horner_SSE2(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                horner_AVX(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                horner_AVX_FMA(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                horner_AVX512F(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
            #elif defined(__aarch64__)
                horner_NEON(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
            #endif
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

            FT_TIME(horner_default(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for horner_default \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(horner_SSE2(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_SSE2 \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(horner_AVX(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_AVX \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(horner_AVX_FMA(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_AVX_FMA \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(horner_AVX512F(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_AVX512F \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(horner_NEON(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for horner_NEON \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #endif

            FT_TIME(ft_horner(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for ft_horner \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

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
            #if defined(__i386__) || defined(__x86_64__)
                clenshaw_SSEf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                clenshaw_AVXf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                clenshaw_AVX_FMAf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                clenshaw_AVX512Ff(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
            #elif defined(__aarch64__)
                clenshaw_NEONf(n, c, 1, m, x, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
            #endif
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

            FT_TIME(clenshaw_defaultf(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for clenshaw_defaultf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(clenshaw_SSEf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_SSEf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(clenshaw_AVXf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_AVXf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(clenshaw_AVX_FMAf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_AVX_FMAf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(clenshaw_AVX512Ff(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_AVX512Ff \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(clenshaw_NEONf(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_NEONf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #endif

            FT_TIME(ft_clenshawf(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for ft_clenshawf \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

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
            #if defined(__i386__) || defined(__x86_64__)
                clenshaw_SSE2(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                clenshaw_AVX(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                clenshaw_AVX_FMA(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                clenshaw_AVX512F(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
            #elif defined(__aarch64__)
                clenshaw_NEON(n, c, 1, m, x, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
            #endif
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

            FT_TIME(clenshaw_default(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for clenshaw_default \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(clenshaw_SSE2(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_SSE2 \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(clenshaw_AVX(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_AVX \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(clenshaw_AVX_FMA(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_AVX_FMA \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(clenshaw_AVX512F(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_AVX512F \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(clenshaw_NEON(n, c, 1, m, x, f), start, end, NTIMES)
                printf("Time for clenshaw_NEON \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #endif

            FT_TIME(ft_clenshaw(n, c, 1, m, x, f), start, end, NTIMES)
            printf("Time for ft_clenshaw \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

            free(c);
            free(x);
            free(f);
        }
    }

    printf("\nTesting methods for Legendre polynomial summation.\n");
    printf("\n\tSingle precision.\n\n");

    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 256; n < 257; n++) {
        for (int m = 256; m < 264; m++) {
            float * c = malloc(n*sizeof(float));
            float * A = malloc(n*sizeof(float));
            float * B = malloc(n*sizeof(float));
            float * C = malloc((n+1)*sizeof(float));
            float * x = malloc(m*sizeof(float));
            float * phi0 = malloc(m*sizeof(float));
            float * f = malloc(m*sizeof(float));
            float * fd = malloc(m*sizeof(float));
            float err = 0.0f;
            for (int k = 0; k < n; k++) {
                c[k] = 1.0f/(k+1.0f);
                A[k] = (2*k+1.0f)/(k+1.0f);
                B[k] = 0.0f;
                C[k] = k/(k+1.0f);
            }
            C[n] = n/(n+1.0f);
            for (int j = 0; j < m; j++) {
                x[j] = (-1.0f+2.0f*j/(m-1.0f));
                phi0[j] = 1.0f;
            }

            orthogonal_polynomial_clenshaw_defaultf(n, c, 1, A, B, C, m, x, phi0, fd);
            #if defined(__i386__) || defined(__x86_64__)
                orthogonal_polynomial_clenshaw_SSEf(n, c, 1, A, B, C, m, x, phi0, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                orthogonal_polynomial_clenshaw_AVXf(n, c, 1, A, B, C, m, x, phi0, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                orthogonal_polynomial_clenshaw_AVX_FMAf(n, c, 1, A, B, C, m, x, phi0, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
                orthogonal_polynomial_clenshaw_AVX512Ff(n, c, 1, A, B, C, m, x, phi0, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
            #elif defined(__aarch64__)
                orthogonal_polynomial_clenshaw_NEONf(n, c, 1, A, B, C, m, x, phi0, f);
                err += powf(ft_norm_2argf(f, fd, m), 2);
            #endif
            ft_orthogonal_polynomial_clenshawf(n, c, 1, A, B, C, m, x, phi0, f);
            err += powf(ft_norm_2argf(f, fd, m), 2);
            err = sqrtf(err);
            err /= ft_norm_1argf(fd, m);

            printf("(m×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", m, n, (double) err);
            ft_checktestf(err, n, &checksum);
            free(c);
            free(A);
            free(B);
            free(C);
            free(x);
            free(phi0);
            free(f);
            free(fd);
        }
    }

    for (int n = MN; n < 2*MN; n *= 2) {
        for (int m = MN; m < 2*MN; m *= 2) {
            float * c = malloc(n*sizeof(float));
            float * A = malloc(n*sizeof(float));
            float * B = malloc(n*sizeof(float));
            float * C = malloc((n+1)*sizeof(float));
            float * x = malloc(m*sizeof(float));
            float * phi0 = malloc(m*sizeof(float));
            float * f = malloc(m*sizeof(float));
            float err = 0.0f;
            for (int k = 0; k < n; k++) {
                c[k] = 1.0f/(k+1.0f);
                A[k] = (2*k+1.0f)/(k+1.0f);
                B[k] = 0.0f;
                C[k] = k/(k+1.0f);
            }
            C[n] = n/(n+1.0f);
            for (int j = 0; j < m; j++) {
                x[j] = (-1.0f+2.0f*j/(m-1.0f));
                phi0[j] = 1.0f;
            }

            FT_TIME(orthogonal_polynomial_clenshaw_defaultf(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
            printf("Time for OP clenshaw_defaultf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(orthogonal_polynomial_clenshaw_SSEf(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_SSEf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(orthogonal_polynomial_clenshaw_AVXf(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_AVXf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(orthogonal_polynomial_clenshaw_AVX_FMAf(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_AVX_FMAf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(orthogonal_polynomial_clenshaw_AVX512Ff(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_AVX512Ff \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(orthogonal_polynomial_clenshaw_NEONf(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_NEONf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #endif

            FT_TIME(ft_orthogonal_polynomial_clenshawf(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
            printf("Time for OP ft_clenshawf \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

            free(c);
            free(A);
            free(B);
            free(C);
            free(x);
            free(phi0);
            free(f);
        }
    }

    printf("\n\tDouble precision.\n\n");

    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");
    for (int n = 256; n < 257; n++) {
        for (int m = 256; m < 264; m++) {
            double * c = malloc(n*sizeof(double));
            double * A = malloc(n*sizeof(double));
            double * B = malloc(n*sizeof(double));
            double * C = malloc((n+1)*sizeof(double));
            double * x = malloc(m*sizeof(double));
            double * phi0 = malloc(m*sizeof(double));
            double * f = malloc(m*sizeof(double));
            double * fd = malloc(m*sizeof(double));
            double err = 0.0;
            for (int k = 0; k < n; k++) {
                c[k] = 1.0/(k+1.0);
                A[k] = (2*k+1.0)/(k+1.0);
                B[k] = 0.0;
                C[k] = k/(k+1.0);
            }
            C[n] = n/(n+1.0);
            for (int j = 0; j < m; j++) {
                x[j] = (-1.0+2.0*j/(m-1.0));
                phi0[j] = 1.0;
            }

            orthogonal_polynomial_clenshaw_default(n, c, 1, A, B, C, m, x, phi0, fd);
            #if defined(__i386__) || defined(__x86_64__)
                orthogonal_polynomial_clenshaw_SSE2(n, c, 1, A, B, C, m, x, phi0, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                orthogonal_polynomial_clenshaw_AVX(n, c, 1, A, B, C, m, x, phi0, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                orthogonal_polynomial_clenshaw_AVX_FMA(n, c, 1, A, B, C, m, x, phi0, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
                orthogonal_polynomial_clenshaw_AVX512F(n, c, 1, A, B, C, m, x, phi0, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
            #elif defined(__aarch64__)
                orthogonal_polynomial_clenshaw_NEON(n, c, 1, A, B, C, m, x, phi0, f);
                err += pow(ft_norm_2arg(f, fd, m), 2);
            #endif
            ft_orthogonal_polynomial_clenshaw(n, c, 1, A, B, C, m, x, phi0, f);
            err += pow(ft_norm_2arg(f, fd, m), 2);
            err = sqrt(err);
            err /= ft_norm_1arg(fd, m);

            printf("(m×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", m, n, (double) err);
            ft_checktest(err, n, &checksum);
            free(c);
            free(A);
            free(B);
            free(C);
            free(x);
            free(phi0);
            free(f);
            free(fd);
        }
    }

    for (int n = MN; n < 2*MN; n *= 2) {
        for (int m = MN; m < 2*MN; m *= 2) {
            double * c = malloc(n*sizeof(double));
            double * A = malloc(n*sizeof(double));
            double * B = malloc(n*sizeof(double));
            double * C = malloc((n+1)*sizeof(double));
            double * x = malloc(m*sizeof(double));
            double * phi0 = malloc(m*sizeof(double));
            double * f = malloc(m*sizeof(double));
            double err = 0.0;
            for (int k = 0; k < n; k++) {
                c[k] = 1.0/(k+1.0);
                A[k] = (2*k+1.0)/(k+1.0);
                B[k] = 0.0;
                C[k] = k/(k+1.0);
            }
            C[n] = n/(n+1.0);
            for (int j = 0; j < m; j++) {
                x[j] = (-1.0+2.0*j/(m-1.0));
                phi0[j] = 1.0;
            }

            FT_TIME(orthogonal_polynomial_clenshaw_default(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
            printf("Time for OP clenshaw_default \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            //for (int i = 0; i < n; i++)
            //    printf("c[%i] = %3.5f, x[%i] = %3.5f, f[%i] = %3.8f\n", i, c[i], i, x[i], i, f[i]);

            #if defined(__i386__) || defined(__x86_64__)
                FT_TIME(orthogonal_polynomial_clenshaw_SSE2(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_SSE2 \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(orthogonal_polynomial_clenshaw_AVX(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_AVX \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(orthogonal_polynomial_clenshaw_AVX_FMA(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_AVX_FMA \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

                FT_TIME(orthogonal_polynomial_clenshaw_AVX512F(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_AVX512F \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #elif defined(__aarch64__)
                FT_TIME(orthogonal_polynomial_clenshaw_NEON(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
                printf("Time for OP clenshaw_NEON \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));
            #endif

            FT_TIME(ft_orthogonal_polynomial_clenshaw(n, c, 1, A, B, C, m, x, phi0, f), start, end, NTIMES)
            printf("Time for OP ft_clenshaw \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NTIMES));

            free(c);
            free(A);
            free(B);
            free(C);
            free(x);
            free(phi0);
            free(f);
        }
    }
    printf("\n");
    return checksum;
}
