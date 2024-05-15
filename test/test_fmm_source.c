#include "fmm_source.h"

void Y(test_fmm)(size_t N, size_t maxs, size_t M, FLT m, size_t random, size_t lagrange,
                 size_t verbose)
{
  if (verbose > 1) printf("test_forward_backward\n");
  X(fmm_plan) *fmmplan = X(create_fmm)(N, maxs, M, BOTH, lagrange, verbose);
  FLT *input_array = (FLT *)calloc(N, sizeof(FLT));
  FLT *output_array = (FLT *)calloc(N, sizeof(FLT));

  // srand((unsigned) time(&t));
  srand(1);
  // Initialize some input array
  if (random == 1)
  {
    for (size_t i = 0; i < N; i++)
      input_array[i] = (2.0*(((double) rand())/RAND_MAX)-1.0) / pow(i + 1, m);
  }
  else
  {
    for (size_t i = 0; i < N; i++)
      input_array[i] = 0.5 / pow(i + 1, m);
  }

  // Leg to Cheb
  if (verbose > 1) printf("Leg2cheb\n");

  size_t flops = X(execute)(input_array, output_array, fmmplan, L2C, 1);

  FLT *ia = (FLT *)calloc(N, sizeof(FLT));
  // Cheb to Leg
  flops += X(execute)(output_array, ia, fmmplan, C2L, 1);

  // Compute maximum error
  FLT error = 0.0;
  for (size_t j = 0; j < N; j++)
  {
    error = MAX(Y(fabs)(input_array[j] - ia[j]), error);
  }

  FLT e0 = 0;
  for (size_t j = 0; j < N; j++)
  {
    e0 = MAX(Y(fabs)(ia[j]), e0);
  }
  FLT ulp;
#if defined(FT_USE_SINGLE)
  ulp = nextafterf(e0, 1e8) - e0;
#elif defined(FT_USE_DOUBLE)
  ulp = nextafter(e0, 1e8) - e0;
#endif

  printf("N %6lu L inf Error = %2.8e \n", N, error / e0);
  printf("               Flops = %lu\n", flops);
  printf("               ulp   = %2.16e\n", ulp);
  printf("          error ulp  = %2.1f\n", error / ulp);
#ifdef TEST
  assert(error < 1e-10);
#endif
  free(ia);
  free(input_array);
  free(output_array);
  X(free_fmm)(fmmplan);
}

void Y(test_fmm_speed)(size_t N, size_t maxs, size_t repeat, size_t direction, size_t M,
                       size_t lagrange, size_t verbose)
{
  if (verbose > 1) printf("test_fmm_speed %lu\n", direction);
  X(fmm_plan) *fmmplan = X(create_fmm)(N, maxs, M, direction, lagrange, verbose);

  FLT *input_array = (FLT *)calloc(N, sizeof(FLT));
  FLT *output_array = (FLT *)calloc(N, sizeof(FLT));
  // Initialize some input array
  for (size_t i = 0; i < N; i++)
    input_array[i] = 1.0;

  uint64_t t0 = tic;
  FLT min_time = 1e8;
  size_t flops;
  for (size_t i = 0; i < repeat; i++)
  {
    for (size_t j = 0; j < N; j++)
      output_array[j] = 0.0;
    uint64_t g0 = tic;
    flops = X(execute)(input_array, output_array, fmmplan, direction, 1);
    FLT s1 = toc(g0);
    min_time = s1 < min_time ? s1 : min_time;
  }

  uint64_t t1 = tic;
  printf("Timing N %6lu avg / min = %2.4e / %2.4e flops = %lu\n", N, dtics(t0, t1) / repeat,
         min_time, flops);
  free(input_array);
  free(output_array);
  X(free_fmm)(fmmplan);
}

void Y(test_fmm_transforms)(int *checksum, int N)
{
  FLT err;
  size_t flops;
#if defined(FT_USE_SINGLE)
  size_t s = 32;
  size_t M = 10;
#elif defined(FT_USE_DOUBLE)
  size_t s = 64;
  size_t M = 18;
#endif

  printf("\nTesting the accuracy of FMM Chebyshev--Legendre transforms (1D).\n\n");
  printf("\t\t\t Test \t\t\t\t | inf-norm Maximum error\n");
  printf("---------------------------------------------------------|----------------------\n");
  for (int n = 64; n < N; n *= 2)
  {
    FLT m = 0;
    FLT *input_array = (FLT *)calloc(n, sizeof(FLT));
    FLT *output_array = (FLT *)calloc(n, sizeof(FLT));
    FLT *ia = (FLT *)calloc(n, sizeof(FLT));
    srand(1);
    FLT e0 = 0;
    for (int i = 0; i < n; i++)
    {
      // input_array[i] = 1;
      input_array[i] = (2.0*(((double) rand())/RAND_MAX)-1.0) / pow(i + 1, m);
      e0 = MAX(e0, Y(fabs)(input_array[i]));
    }

    X(fmm_plan) *fmmplan = X(create_fmm)(n, s, M, BOTH, 0, 0);
    flops = X(execute)(input_array, output_array, fmmplan, L2C, 1);
    flops += X(execute)(output_array, ia, fmmplan, C2L, 1);
    X(free_fmm)(fmmplan);
    err = X(normInf_2arg)(ia, input_array, n) / e0;
    printf("n = %5i: \t\t\t\t\t\t |%20.2e ", n, (double)err);
    X(checktest)(err, 2 * Y(sqrt)(n), checksum);
    free(input_array);
    free(output_array);
    free(ia);
  }
}
void Y(test_fmm_transforms2d)(int *checksum, int N)
{
  FLT err;
  size_t flops;
#if defined(FT_USE_SINGLE)
  size_t s = 32;
  size_t M = 10;
#elif defined(FT_USE_DOUBLE)
  size_t s = 64;
  size_t M = 18;
#endif

  printf("\nTesting the accuracy of FMM Chebyshev--Legendre transforms (2D).\n\n");
  printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
  printf("---------------------------------------------------------|----------------------\n");
  for (int n = 64; n < N; n *= 2)
  {
    FLT *input_array = (FLT *)calloc(n*n, sizeof(FLT));
    FLT *output_array = (FLT *)calloc(n*n, sizeof(FLT));
    FLT *ia = (FLT *)calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
    {
      input_array[i+i*n] = 1;
    }

    X(fmm_plan_2d) *fmmplan = X(create_fmm_2d)(n, n, 0, s, M, BOTH, 0, 0);
    flops = X(execute2D)(input_array, output_array, fmmplan, L2C);
    flops += X(execute2D)(output_array, ia, fmmplan, C2L);
    X(free_fmm_2d)(fmmplan);
    err = X(norm_2arg)(ia, input_array, n*n) / X(norm_1arg)(ia, n*n);
    printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, (double)err);
    X(checktest)(err, 2 * Y(sqrt)(n), checksum);
    free(input_array);
    free(output_array);
    free(ia);
  }
}
