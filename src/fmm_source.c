#include "fmm_source.h"

// C ← α*A*B + β*C, C ← α*Aᵀ*B + β*C
#if defined(FT_USE_SINGLE)
// y ← α*A*x + β*y, y ← α*Aᵀ*x + β*y
void X(rm_gemv)(char TRANS, int m, int n, FLT alpha, FLT *A, int LDA, FLT *x, FLT beta, FLT *y)
{
  if (TRANS == 'N')
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, LDA, x, 1, beta, y, 1);
  else if (TRANS == 'T')
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, alpha, A, LDA, x, 1, beta, y, 1);
}

void X(rm_gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT *A, int LDA, FLT *B, int LDB,
                FLT beta, FLT *C, int LDC)
{
  if (TRANS == 'N')
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, A, LDA, B, LDB, beta, C,
                LDC);
  else if (TRANS == 'T')
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, p, alpha, A, LDA, B, LDB, beta, C,
                LDC);
}

FLT *X(fftw_malloc)(int N) { return (FLT *)fftwf_malloc(N); }

typedef fftwf_plan X(fftw_plan);

fftwf_plan X(fftw_plan_r2r_1d)(int M, FLT *fun, FLT *fun_hat)
{
  return fftwf_plan_r2r_1d(M, fun, fun_hat, FFTW_REDFT10, FFTW_MEASURE);
}

fftwf_plan X(fftw_plan_r2r_2d)(int M, FLT *fun, FLT *fun_hat)
{
  return fftwf_plan_r2r_2d(M, M, fun, fun_hat, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
}

void X(fftw_execute)(fftwf_plan plan) { fftwf_execute(plan); }

void X(fftw_destroy_plan)(fftwf_plan plan) { fftwf_destroy_plan(plan); }

#elif defined(FT_USE_DOUBLE)

void X(rm_gemv)(char TRANS, int m, int n, FLT alpha, FLT *A, int LDA, FLT *x, FLT beta, FLT *y)
{
  if (TRANS == 'N')
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, LDA, x, 1, beta, y, 1);
  else if (TRANS == 'T')
    cblas_dgemv(CblasRowMajor, CblasTrans, n, m, alpha, A, LDA, x, 1, beta, y, 1);
}

void X(rm_gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT *A, int LDA, FLT *B, int LDB,
                FLT beta, FLT *C, int LDC)
{
  if (TRANS == 'N')
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, A, LDA, B, LDB, beta, C,
                LDC);
  else if (TRANS == 'T')
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, p, alpha, A, LDA, B, LDB, beta, C,
                LDC);
}

FLT *X(fftw_malloc)(int N) { return (FLT *)fftw_malloc(N); }

typedef fftw_plan X(fftw_plan);

fftw_plan X(fftw_plan_r2r_1d)(int M, FLT *fun, FLT *fun_hat)
{
  return fftw_plan_r2r_1d(M, fun, fun_hat, FFTW_REDFT10, FFTW_MEASURE);
}

fftw_plan X(fftw_plan_r2r_2d)(int M, FLT *fun, FLT *fun_hat)
{
  return fftw_plan_r2r_2d(M, M, fun, fun_hat, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
}

void X(fftw_execute)(fftw_plan plan) { fftw_execute(plan); }

void X(fftw_destroy_plan)(fftw_plan plan) { fftw_destroy_plan(plan); }

// Compact storage of lower triangular transform matrix. (18 x 18 + 18) / 2
// items
const double BMe[171] __attribute__((aligned)) = {
    1.0000000000000000e+00,  -5.0000000000000000e-01, 5.0000000000000000e-01,
    -2.5000000000000000e-01, -1.0000000000000000e+00, 2.5000000000000000e-01,
    2.5000000000000000e-01,  3.7500000000000000e-01,  -7.5000000000000000e-01,
    1.2500000000000000e-01,  1.8750000000000000e-01,  5.0000000000000000e-01,
    7.5000000000000000e-01,  -5.0000000000000000e-01, 6.2500000000000000e-02,
    -1.8750000000000000e-01, -3.1250000000000000e-01, 0.0000000000000000e+00,
    7.8125000000000000e-01,  -3.1250000000000000e-01, 3.1250000000000000e-02,
    -1.5625000000000000e-01, -3.7500000000000000e-01, -5.1562500000000000e-01,
    -4.3750000000000000e-01, 6.5625000000000000e-01,  -1.8750000000000000e-01,
    1.5625000000000000e-02,  1.5625000000000000e-01,  2.7343750000000000e-01,
    1.0937500000000000e-01,  -2.7343750000000000e-01, -6.5625000000000000e-01,
    4.9218750000000000e-01,  -1.0937500000000000e-01, 7.8125000000000000e-03,
    1.3671875000000000e-01,  3.1250000000000000e-01,  4.0625000000000000e-01,
    4.3750000000000000e-01,  1.0937500000000000e-01,  -6.8750000000000000e-01,
    3.4375000000000000e-01,  -6.2500000000000000e-02, 3.9062500000000000e-03,
    -1.3671875000000000e-01, -2.4609375000000000e-01, -1.4062500000000000e-01,
    9.3750000000000000e-02,  4.2187500000000000e-01,  4.2187500000000000e-01,
    -6.0937500000000000e-01, 2.2851562500000000e-01,  -3.5156250000000000e-02,
    1.9531250000000000e-03,  -1.2304687500000000e-01, -2.7343750000000000e-01,
    -3.4179687500000000e-01, -3.9062500000000000e-01, -2.7343750000000000e-01,
    1.7187500000000000e-01,  5.9082031250000000e-01,  -4.8828125000000000e-01,
    1.4648437500000000e-01,  -1.9531250000000000e-02, 9.7656250000000000e-04,
    1.2304687500000000e-01,  2.2558593750000000e-01,  1.5039062500000000e-01,
    -1.0742187500000000e-02, -2.5781250000000000e-01, -4.3505859375000000e-01,
    -1.3964843750000000e-01, 6.2841796875000000e-01,  -3.6523437500000000e-01,
    9.1308593750000000e-02,  -1.0742187500000000e-02, 4.8828125000000000e-04,
    1.1279296875000000e-01,  2.4609375000000000e-01,  2.9882812500000000e-01,
    3.4765625000000000e-01,  3.0834960937500000e-01,  6.4453125000000000e-02,
    -3.5449218750000000e-01, -3.9257812500000000e-01, 5.7861328125000000e-01,
    -2.5976562500000000e-01, 5.5664062500000000e-02,  -5.8593750000000000e-03,
    2.4414062500000000e-04,  -1.1279296875000000e-01, -2.0947265625000000e-01,
    -1.5234375000000000e-01, -3.3325195312500000e-02, 1.5551757812500000e-01,
    3.4753417968750000e-01,  3.3007812500000000e-01,  -1.2377929687500000e-01,
    -5.3955078125000000e-01, 4.8559570312500000e-01,  -1.7773437500000000e-01,
    3.3325195312500000e-02,  -3.1738281250000000e-03, 1.2207031250000000e-04,
    -1.0473632812500000e-01, -2.2558593750000000e-01, -2.6788330078125000e-01,
    -3.1274414062500000e-01, -3.0676269531250000e-01, -1.6918945312500000e-01,
    1.3629150390625000e-01,  4.1162109375000000e-01,  1.4184570312500000e-01,
    -5.8447265625000000e-01, 3.8153076171875000e-01,  -1.1791992187500000e-01,
    1.9653320312500000e-02,  -1.7089843750000000e-03, 6.1035156250000000e-05,
    1.0473632812500000e-01,  1.9638061523437500e-01,  1.5106201171875000e-01,
    5.8746337890625000e-02,  -8.9721679687500000e-02, -2.6358032226562500e-01,
    -3.4515380859375000e-01, -1.4877319335937500e-01, 3.1127929687500000e-01,
    3.6056518554687500e-01,  -5.5499267578125000e-01, 2.8518676757812500e-01,
    -7.6293945312500000e-02, 1.1444091796875000e-02,  -9.1552734375000000e-04,
    3.0517578125000000e-05,  9.8190307617187500e-02,  2.0947265625000000e-01,
    2.4438476562500000e-01,  2.8466796875000000e-01,  2.9406738281250000e-01,
    2.1533203125000000e-01,  2.6855468750000000e-03,  -2.7978515625000000e-01,
    -3.4722900390625000e-01, 1.0205078125000000e-01,  4.9633789062500000e-01,
    -4.8291015625000000e-01, 2.0495605468750000e-01,  -4.8339843750000000e-02,
    6.5917968750000000e-03,  -4.8828125000000000e-04, 1.5258789062500000e-05,
    -9.8190307617187500e-02, -1.8547058105468750e-01, -1.4837646484375000e-01,
    -7.4188232421875000e-02, 4.5654296875000000e-02,  1.9662475585937500e-01,
    3.1024169921875000e-01,  2.5628662109375000e-01,  -5.2917480468750000e-02,
    -3.8806152343750000e-01, -1.3177490234375000e-01, 5.4837036132812500e-01,
    -3.9428710937500000e-01, 1.4266967773437500e-01,  -3.0090332031250000e-02,
    3.7612915039062500e-03,  -2.5939941406250000e-04, 7.6293945312500000e-06};

// Transpose of BMe
const double BMeT[171] __attribute__((aligned)) = {
    1.0000000000000000e+00,  -5.0000000000000000e-01, -2.5000000000000000e-01,
    2.5000000000000000e-01,  1.8750000000000000e-01,  -1.8750000000000000e-01,
    -1.5625000000000000e-01, 1.5625000000000000e-01,  1.3671875000000000e-01,
    -1.3671875000000000e-01, -1.2304687500000000e-01, 1.2304687500000000e-01,
    1.1279296875000000e-01,  -1.1279296875000000e-01, -1.0473632812500000e-01,
    1.0473632812500000e-01,  9.8190307617187500e-02,  -9.8190307617187500e-02,
    5.0000000000000000e-01,  -1.0000000000000000e+00, 3.7500000000000000e-01,
    5.0000000000000000e-01,  -3.1250000000000000e-01, -3.7500000000000000e-01,
    2.7343750000000000e-01,  3.1250000000000000e-01,  -2.4609375000000000e-01,
    -2.7343750000000000e-01, 2.2558593750000000e-01,  2.4609375000000000e-01,
    -2.0947265625000000e-01, -2.2558593750000000e-01, 1.9638061523437500e-01,
    2.0947265625000000e-01,  -1.8547058105468750e-01, 2.5000000000000000e-01,
    -7.5000000000000000e-01, 7.5000000000000000e-01,  0.0000000000000000e+00,
    -5.1562500000000000e-01, 1.0937500000000000e-01,  4.0625000000000000e-01,
    -1.4062500000000000e-01, -3.4179687500000000e-01, 1.5039062500000000e-01,
    2.9882812500000000e-01,  -1.5234375000000000e-01, -2.6788330078125000e-01,
    1.5106201171875000e-01,  2.4438476562500000e-01,  -1.4837646484375000e-01,
    1.2500000000000000e-01,  -5.0000000000000000e-01, 7.8125000000000000e-01,
    -4.3750000000000000e-01, -2.7343750000000000e-01, 4.3750000000000000e-01,
    9.3750000000000000e-02,  -3.9062500000000000e-01, -1.0742187500000000e-02,
    3.4765625000000000e-01,  -3.3325195312500000e-02, -3.1274414062500000e-01,
    5.8746337890625000e-02,  2.8466796875000000e-01,  -7.4188232421875000e-02,
    6.2500000000000000e-02,  -3.1250000000000000e-01, 6.5625000000000000e-01,
    -6.5625000000000000e-01, 1.0937500000000000e-01,  4.2187500000000000e-01,
    -2.7343750000000000e-01, -2.5781250000000000e-01, 3.0834960937500000e-01,
    1.5551757812500000e-01,  -3.0676269531250000e-01, -8.9721679687500000e-02,
    2.9406738281250000e-01,  4.5654296875000000e-02,  3.1250000000000000e-02,
    -1.8750000000000000e-01, 4.9218750000000000e-01,  -6.8750000000000000e-01,
    4.2187500000000000e-01,  1.7187500000000000e-01,  -4.3505859375000000e-01,
    6.4453125000000000e-02,  3.4753417968750000e-01,  -1.6918945312500000e-01,
    -2.6358032226562500e-01, 2.1533203125000000e-01,  1.9662475585937500e-01,
    1.5625000000000000e-02,  -1.0937500000000000e-01, 3.4375000000000000e-01,
    -6.0937500000000000e-01, 5.9082031250000000e-01,  -1.3964843750000000e-01,
    -3.5449218750000000e-01, 3.3007812500000000e-01,  1.3629150390625000e-01,
    -3.4515380859375000e-01, 2.6855468750000000e-03,  3.1024169921875000e-01,
    7.8125000000000000e-03,  -6.2500000000000000e-02, 2.2851562500000000e-01,
    -4.8828125000000000e-01, 6.2841796875000000e-01,  -3.9257812500000000e-01,
    -1.2377929687500000e-01, 4.1162109375000000e-01,  -1.4877319335937500e-01,
    -2.7978515625000000e-01, 2.5628662109375000e-01,  3.9062500000000000e-03,
    -3.5156250000000000e-02, 1.4648437500000000e-01,  -3.6523437500000000e-01,
    5.7861328125000000e-01,  -5.3955078125000000e-01, 1.4184570312500000e-01,
    3.1127929687500000e-01,  -3.4722900390625000e-01, -5.2917480468750000e-02,
    1.9531250000000000e-03,  -1.9531250000000000e-02, 9.1308593750000000e-02,
    -2.5976562500000000e-01, 4.8559570312500000e-01,  -5.8447265625000000e-01,
    3.6056518554687500e-01,  1.0205078125000000e-01,  -3.8806152343750000e-01,
    9.7656250000000000e-04,  -1.0742187500000000e-02, 5.5664062500000000e-02,
    -1.7773437500000000e-01, 3.8153076171875000e-01,  -5.5499267578125000e-01,
    4.9633789062500000e-01,  -1.3177490234375000e-01, 4.8828125000000000e-04,
    -5.8593750000000000e-03, 3.3325195312500000e-02,  -1.1791992187500000e-01,
    2.8518676757812500e-01,  -4.8291015625000000e-01, 5.4837036132812500e-01,
    2.4414062500000000e-04,  -3.1738281250000000e-03, 1.9653320312500000e-02,
    -7.6293945312500000e-02, 2.0495605468750000e-01,  -3.9428710937500000e-01,
    1.2207031250000000e-04,  -1.7089843750000000e-03, 1.1444091796875000e-02,
    -4.8339843750000000e-02, 1.4266967773437500e-01,  6.1035156250000000e-05,
    -9.1552734375000000e-04, 6.5917968750000000e-03,  -3.0090332031250000e-02,
    3.0517578125000000e-05,  -4.8828125000000000e-04, 3.7612915039062500e-03,
    1.5258789062500000e-05,  -2.5939941406250000e-04, 7.6293945312500000e-06};

// Mixed radix 2 and 3 algorithm
void dct_radix23(double *input, double *output, size_t st)
{
  double out[18], z[9], B[3], C[3], T[9];
  double f[3], g[3], h[3], t[3];

  static const double M_SQRT3_2 = 8.660254037844385966e-01;
  // sqrt(3)*sin(pi*(i+0.5)/9)
  static const double SIN[3] __attribute__((aligned)) = {
      3.007674663608705945e-01, 8.660254037844385966e-01, 1.326827896337876789e+00};
  // sqrt(3)*sin(2*pi*(i+0.5)/9)
  static const double SIN2[3] __attribute__((aligned)) = {
      5.923962654520477100e-01, 1.500000000000000000e+00, 1.705737063904886330e+00};
  // cos(pi*(i+0.5)/9)
  static const double COS[3] __attribute__((aligned)) = {
      9.848077530122080203e-01, 8.660254037844385966e-01, 6.427876096865393629e-01};
  // cos(2*pi*(i+0.5)/9)
  static const double COS2[3] __attribute__((aligned)) = {
      9.396926207859084279e-01, 5.000000000000000000e-01, -1.736481776669303589e-01};
  // 2*cos(pi*(i+0.5)/18)
  static const double COSH[9] __attribute__((aligned)) = {
      1.992389396183491090e+00, 1.931851652578136624e+00, 1.812615574073299873e+00,
      1.638304088577983597e+00, 1.414213562373095145e+00, 1.147152872702092097e+00,
      8.452365234813988826e-01, 5.176380902050414790e-01, 1.743114854953163595e-01};

  // even indices
  for (size_t i = 0; i < 9; i++)
  { // radix 2
    z[i] = input[i * st] + input[(17 - i) * st];
  }

  // radix 3 on the 9 even indices
  for (size_t i = 0; i < 3; i++)
  {
    f[i] = z[i] + z[5 - i] + z[6 + i];
    t[i] = 2 * z[i] - z[5 - i] - z[6 + i];
  }

  for (size_t i = 0; i < 3; i++)
  {
    double tmp = z[5 - i] - z[6 + i];
    g[i] = t[i] * COS[i] + tmp * SIN[i];
    h[i] = t[i] * COS2[i] - tmp * SIN2[i];
  }

  B[0] = g[0] + g[1] + g[2];
  B[1] = M_SQRT3_2 * (g[0] - g[2]);
  B[2] = 0.5 * (g[0] + g[2]) - g[1];
  C[0] = h[0] + h[1] + h[2];
  C[1] = M_SQRT3_2 * (h[0] - h[2]);
  C[2] = 0.5 * (h[0] + h[2]) - h[1];
  out[0] = f[0] + f[1] + f[2];
  out[3] = M_SQRT3_2 * (f[0] - f[2]);
  out[6] = 0.5 * (f[0] + f[2]) - f[1];
  out[1] = B[0] * 0.5;
  out[2] = C[0] * 0.5;
  out[4] = B[1] - out[2];
  out[5] = C[1] - out[1];
  out[7] = B[2] - out[5];
  out[8] = C[2] - out[4];

  // odd indices
  for (size_t i = 0; i < 9; i++)
  {
    z[i] = (input[i * st] - input[(17 - i) * st]) * COSH[i];
  }

  // radix 3 on the 9 odd indices
  for (size_t i = 0; i < 3; i++)
  {
    f[i] = z[i] + z[5 - i] + z[6 + i];
  }
  out[9] = (f[0] + f[1] + f[2]) * 0.5; // T[0] is out[9]
  T[3] = M_SQRT3_2 * (f[0] - f[2]);
  T[6] = 0.5 * (f[0] + f[2]) - f[1];

  for (size_t i = 0; i < 3; i++)
  {
    t[i] = 3 * z[i] - f[i];
  }

  for (size_t i = 0; i < 3; i++)
  {
    double tmp = z[5 - i] - z[6 + i];
    g[i] = t[i] * COS[i] + tmp * SIN[i];
    h[i] = t[i] * COS2[i] - tmp * SIN2[i];
  }

  B[0] = g[0] + g[1] + g[2];
  B[1] = M_SQRT3_2 * (g[0] - g[2]);
  B[2] = 0.5 * (g[0] + g[2]) - g[1];
  C[0] = h[0] + h[1] + h[2];
  C[1] = M_SQRT3_2 * (h[0] - h[2]);
  C[2] = 0.5 * (h[0] + h[2]) - h[1];
  T[1] = B[0] * 0.5;
  T[2] = C[0] * 0.5;
  T[4] = B[1] - T[2];
  T[5] = C[1] - T[1];
  T[7] = B[2] - T[5];
  T[8] = C[2] - T[4];

  for (size_t i = 1; i < 9; i++)
  {
    out[9 + i] = T[i] - out[8 + i];
  }

  for (size_t i = 0; i < 9; i++)
  {
    output[2 * i * st] = out[i];
    output[(2 * i + 1) * st] = out[9 + i];
  }
}

void dct2(double *input, double *output)
{
  double tmp[324];
  static const double sc = 1.0 / 81.0;
  for (size_t i = 0; i < 18; i++)
  {
    dct_radix23(input + i * 18, tmp + i * 18, 1);
  }
  for (size_t i = 0; i < 18; i++)
  {
    dct_radix23(tmp + i, output + i, 18);
  }
  for (size_t i = 0; i < 324; i++)
  {
    output[i] *= sc;
  }
  for (size_t i = 0; i < 18; i++)
  {
    output[i] *= 0.5;
    output[i * 18] *= 0.5;
  }
}

#else

void X(rm_gemv)(char TRANS, int m, int n, FLT alpha, FLT *A, int LDA, FLT *x, FLT beta, FLT *y)
{
  if (TRANS == 'N')
    X(gemv)('T', m, n, alpha, A, LDA, x, beta, y);
  else
    X(gemv)('N', m, n, alpha, A, LDA, x, beta, y);
}

void X(rm_gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT *A, int LDA, FLT *B, int LDB,
                FLT beta, FLT *C, int LDC)
{
  if (TRANS == 'T')
    X(gemm)('N', m, n, p, alpha, A, LDA, B, LDB, beta, C, LDC);
  else
    X(gemm)('T', m, n, p, alpha, A, LDA, B, LDB, beta, C, LDC);
}

FLT *X(fftwl_malloc)(int N) { return (FLT *)fftwl_malloc(N); }

typedef fftwl_plan X(fftw_plan);

fftwl_plan X(fftw_plan_r2r_1d)(int M, FLT *fun, FLT *fun_hat)
{
  return fftwl_plan_r2r_1d(M, fun, fun_hat, FFTW_REDFT10, FFTW_MEASURE);
}

fftwl_plan X(fftw_plan_r2r_2d)(int M, FLT *fun, FLT *fun_hat)
{
  return fftwl_plan_r2r_2d(M, M, fun, fun_hat, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
}

void X(fftw_execute)(fftwl_plan plan) { fftwl_execute(plan); }

void X(fftw_destroy_plan)(fftwl_plan plan) { fftwl_destroy_plan(plan); }

#endif

FLT X(_Lambda0)(const FLT z) { return exp(lgamma(z + 0.5) - lgamma(z + 1)); }

FLT X(Lambda)(const FLT z)
{
  FLT zz, y;
  const FLT zp = z + 0.25;
  const FLT s = 1 / sqrt(zp);
  const FLT z2 = 1 / (zp * zp);

  if (z < 10) return X(Lambda)(z + 1) * (z + 1) / (z + 0.5);

  y = 1 - 1.5625e-02 * z2;
  if (z > 2014) // 2000
    goto scalereturn;
  zz = z2 * z2;
  y += 2.5634765625e-03 * zz;
  if (z > 141) // 92
    goto scalereturn;
  zz *= z2;
  y -= 1.2798309326171875e-03 * zz;
  if (z > 40) // 31
    goto scalereturn;
  zz *= z2;
  y += 1.343511044979095458984375e-03 * zz;
  if (z > 20) goto scalereturn;
  zz *= z2;
  y -= 2.4328966392204165e-03 * zz;
  if (z > 14) goto scalereturn;
  zz *= z2;
  y += 6.754237533641572e-03 * zz;
  goto scalereturn;
scalereturn:
  return y * s;
}

// Lambda with integer argument. For initializing direct solver
FLT X(LambdaI)(const size_t z)
{
  const static FLT LamInt[256] = {
      1.7724538509055161e+00, 8.8622692545275805e-01, 6.6467019408956851e-01,
      5.5389182840797380e-01, 4.8465534985697706e-01, 4.3618981487127934e-01,
      3.9984066363200604e-01, 3.7128061622971992e-01, 3.4807557771536241e-01,
      3.2873804562006448e-01, 3.1230114333906128e-01, 2.9810563682364938e-01,
      2.8568456862266400e-01, 2.7469670059871537e-01, 2.6488610414876129e-01,
      2.5605656734380255e-01, 2.4805479961430874e-01, 2.4075907021388790e-01,
      2.3407131826350211e-01, 2.2791154673025205e-01, 2.2221375806199575e-01,
      2.1692295429861491e-01, 2.1199288715546458e-01, 2.0738434613034576e-01,
      2.0306383891929691e-01, 1.9900256214091097e-01, 1.9517558979204730e-01,
      1.9156122701812048e-01, 1.8814049082136833e-01, 1.8489668925548267e-01,
      1.8181507776789130e-01, 1.7888257651357048e-01, 1.7608753625554593e-01,
      1.7341954328197706e-01, 1.7086925588077151e-01, 1.6842826651104620e-01,
      1.6608898503172612e-01, 1.6384453928805415e-01, 1.6168869008689554e-01,
      1.5961575816270457e-01, 1.5762056118567075e-01, 1.5569835921999184e-01,
      1.5384480732451575e-01, 1.5205591421609116e-01, 1.5032800609999922e-01,
      1.4865769492111033e-01, 1.4704185041109827e-01, 1.4547757540672487e-01,
      1.4396218399623817e-01, 1.4249318211872553e-01, 1.4106825029753828e-01,
      1.3968522823579768e-01, 1.3834210104122271e-01, 1.3703698688045646e-01,
      1.3576812589082260e-01, 1.3453387020090604e-01, 1.3333267493125509e-01,
      1.3216309006343707e-01, 1.3102375308013156e-01, 1.2991338229131691e-01,
      1.2883077077222260e-01, 1.2777478084786012e-01, 1.2674433906682897e-01,
      1.2573843161391765e-01, 1.2475610011693392e-01, 1.2379643780834211e-01,
      1.2285858600676376e-01, 1.2194173088731031e-01, 1.2104510051313890e-01,
      1.2016796210362340e-01, 1.1930961951716895e-01, 1.1846941092901987e-01,
      1.1764670668645723e-01, 1.1684090732559109e-01, 1.1605144173555330e-01,
      1.1527776545731629e-01, 1.1451935910562341e-01, 1.1377572690363885e-01,
      1.1304639532092321e-01, 1.1233091180623384e-01, 1.1162884360744486e-01,
      1.1093977667159645e-01, 1.1026331461872085e-01, 1.0959907778366831e-01,
      1.0894670232067029e-01, 1.0830583936584282e-01, 1.0767615425325071e-01,
      1.0705732578053088e-01, 1.0644904552041423e-01, 1.0585101717479392e-01,
      1.0526295596826729e-01, 1.0468458807833175e-01, 1.0411565009964517e-01,
      1.0355588853996966e-01, 1.0300505934560812e-01, 1.0246292745431544e-01,
      1.0192926637382421e-01, 1.0140385778426843e-01, 1.0088649116292012e-01,
      1.0037696342977405e-01, 9.9875078612625179e-02, 9.9380647530384461e-02,
      9.8893487493470808e-02, 9.8413422020201535e-02, 9.7940280568181340e-02,
      9.7473898279761426e-02, 9.7014115740705953e-02, 9.6560778751263399e-02,
      9.6113738108896438e-02, 9.5672849401974888e-02, 9.5237972813784100e-02,
      9.4808972936244532e-02, 9.4385718592779153e-02, 9.3968082669802250e-02,
      9.3555941956338207e-02, 9.3149176991310659e-02, 9.2747671918072247e-02,
      9.2351314345772789e-02, 9.1959995217189006e-02, 9.1573608682663010e-02,
      9.1192051979818570e-02, 9.0815225318744947e-02, 9.0443031772356644e-02,
      9.0075377171656007e-02, 8.9712170005641273e-02, 8.9353321325618698e-02,
      8.8998744653691647e-02, 8.8648355895212541e-02, 8.8302073254996866e-02,
      8.7959817157109280e-02, 8.7621510168043482e-02, 8.7287076923127288e-02,
      8.6956444055994217e-02, 8.6629540130971683e-02, 8.6306295578244180e-02,
      8.5986642631658089e-02, 8.5670515269041708e-02, 8.5357849154921117e-02,
      8.5048581585519228e-02, 8.4742651435931030e-02, 8.4439999109374136e-02,
      8.4140566488418903e-02, 8.3844296888107572e-02, 8.3551135010876423e-02,
      8.3261026903199767e-02, 8.2973919913878397e-02, 8.2689762653899351e-02,
      8.2408504957797654e-02, 8.2130097846453740e-02, 8.1854493491264307e-02,
      8.1581645179626752e-02, 8.1311507281680975e-02, 8.1044035218254387e-02,
      8.0779185429959446e-02, 8.0516915347394635e-02, 8.0257183362403048e-02,
      7.9999948800344056e-02, 7.9745171893336589e-02, 7.9492813754433622e-02,
      7.9242836352690124e-02, 7.8995202489087965e-02, 7.8749875773283351e-02,
      7.8506820601143584e-02, 7.8266002133041912e-02, 7.8027386272880209e-02,
      7.7790939647810864e-02, 7.7556629588630716e-02, 7.7324424110820439e-02,
      7.7094291896204911e-02, 7.6866202275210224e-02, 7.6640125209694890e-02,
      7.6416031276333216e-02, 7.6193891650529921e-02, 7.5973678090846306e-02,
      7.5755362923918587e-02, 7.5538919029850243e-02, 7.5324319828060898e-02,
      7.5111539263574847e-02, 7.4900551793733353e-02, 7.4691332375315098e-02,
      7.4483856452050343e-02, 7.4278099942514289e-02, 7.4074039228386498e-02,
      7.3871651143063044e-02, 7.3670912960609070e-02, 7.3471802385039850e-02,
      7.3274297539918778e-02, 7.3078376958261235e-02, 7.2884019572733952e-02,
      7.2691204706139420e-02, 7.2499912062175889e-02, 7.2310121716463394e-02,
      7.2121814107826768e-02, 7.1934970029827211e-02, 7.1749570622533843e-02,
      7.1565597364527347e-02, 7.1383032065128041e-02, 7.1201856856840912e-02,
      7.1022054188010511e-02, 7.0843606815678820e-02, 7.0666497798639621e-02,
      7.0490710490682812e-02, 7.0316228534022709e-02, 7.0143035852904420e-02,
      6.9971116647382606e-02, 6.9800455387267035e-02, 6.9631036806229979e-02,
      6.9462845896070005e-02, 6.9295867901127531e-02, 6.9130088312847310e-02,
      6.8965492864483391e-02, 6.8802067525941965e-02, 6.8639798498758134e-02,
      6.8478672211202365e-02, 6.8318675313512642e-02, 6.8159794673248661e-02,
      6.8002017370764292e-02, 6.7845330694794787e-02, 6.7689722138155342e-02,
      6.7535179393547681e-02, 6.7381690349471446e-02, 6.7229243086237345e-02,
      6.7077825872079153e-02, 6.6927427159361480e-02, 6.6778035580880774e-02,
      6.6629639946256591e-02, 6.6482229238410892e-02, 6.6335792610132449e-02,
      6.6190319380724269e-02, 6.6045799032731417e-02, 6.5902221208747211e-02,
      6.5759575708295381e-02, 6.5617852484786132e-02, 6.5477041642544101e-02,
      6.5337133433906180e-02, 6.5198118256387230e-02, 6.5059986649911833e-02,
      6.4922729294110332e-02, 6.4786337005677333e-02, 6.4650800735790978e-02,
      6.4516111567591405e-02, 6.4382260713716735e-02, 6.4249239513895010e-02,
      6.4117039432590700e-02, 6.3985652056704242e-02, 6.3855069093323211e-02,
      6.3725282367523783e-02, 6.3596283820221103e-02, 6.3468065506067428e-02,
      6.3340619591396613e-02, 6.3213938352213811e-02, 6.3088014172229326e-02,
      6.2962839540935220e-02, 6.2838407051723888e-02, 6.2714709400047267e-02,
      6.2591739381615802e-02};
  if (z > 255) return X(Lambda)((const FLT)z);
  return LamInt[z];
}

size_t X(directM)(const FLT *input_array, FLT *output_array, X(fmm_plan) * fmmplan,
                  const size_t strides)
{
  size_t s = fmmplan->s;
  size_t N = fmmplan->N;
  size_t flops = 0;
  size_t h = 2 * s;
  size_t nL = N / h;
  const FLT *a = fmmplan->dplan->a;

  for (size_t block = 0; block < nL - 1; block++)
  {
    size_t i0 = block * h;
    for (size_t n = 0; n < 4 * s; n = n + 2)
    {
      const size_t n1 = n / 2;
      const FLT *ap = &a[i0 + n1];
      const FLT a0 = a[n1];
      size_t i;
      if (strides == 1)
      {
        FLT *vp = &output_array[i0];
        const FLT *up = &input_array[i0 + n];
        if (n < h)
        {
          for (i = 0; i < h; i++)
          {
            vp[i] += a0 * ap[i] * up[i];
          }
        }
        else
        {
          for (i = 0; i < 2 * h - n; i++)
          {
            vp[i] += a0 * ap[i] * up[i];
          }
        }
      }
      else
      {
        FLT *vp = &output_array[i0 * strides];
        const FLT *up = &input_array[(i0 + n) * strides];
        if (n < h)
        {
          for (i = 0; i < h; i++)
          {
            vp[i * strides] += a0 * ap[i] * up[i * strides];
          }
        }
        else
        {
          for (i = 0; i < 2 * h - n; i++)
          {
            vp[i * strides] += a0 * ap[i] * up[i * strides];
          }
        }
      }
      flops += i * 3;
    }
  }

  // Last block
  size_t i0 = (nL - 1) * h;
  for (size_t n = 0; n < N - i0; n++)
  {
    const FLT *ap = &a[n + i0];
    const FLT a0 = a[n];
    const FLT *cp = &input_array[(i0 + 2 * n) * strides];
    FLT *op = &output_array[i0 * strides];
    if ((int)N - (2 * n + i0) <= 0) break;
    if (strides == 1)
    {
      for (int i = 0; i < N - (2 * n + i0); i++)
      {
        op[i] += a0 * ap[i] * cp[i];
      }
    }
    else
    {
      for (int i = 0; i < N - (2 * n + i0); i++)
      {
        op[i * strides] += a0 * ap[i] * cp[i * strides];
      }
    }
    flops += (N - (2 * n + i0)) * 3;
  }

  FLT *op = &output_array[0];
  if (strides == 1)
  {
    {
      for (size_t i = 0; i < N; i++)
        output_array[i] *= M_2_PI;
    }
  }
  else
  {
    for (size_t i = 0; i < N; i++)
    {
      (*op) *= M_2_PI;
      op += strides;
    }
  }
  output_array[0] *= 0.5;
  return flops;
}

size_t X(directL)(const FLT *input, FLT *output_array, X(fmm_plan) * fmmplan, size_t strides)
{
  const FLT sqrt_pi = 1.77245385090551602729816e0;
  size_t s = fmmplan->s;
  size_t N = fmmplan->N;
  const FLT *an = fmmplan->dplan->an;
  const FLT *dn = fmmplan->dplan->dn;
  size_t h = 2 * s;
  size_t nL = N / h;
  size_t flops = 0;
  FLT *op = &output_array[0];
  const FLT *ia = &input[0];
  const FLT *ap = &an[0];
  if (strides == 1)
  {
    for (size_t i = 0; i < N; i++)
      (*op++) += sqrt_pi * (*ia++) * (*ap++);
  }
  else
  {
    for (size_t i = 0; i < N; i++)
    {
      (*op) += sqrt_pi * (*ia++) * (*ap++);
      op += strides;
    }
  }
  flops += N * 3;

  for (size_t block = 0; block < nL - 1; block++)
  {
    size_t i0 = block * h;
    for (size_t n = 2; n < 4 * s; n = n + 2)
    {
      const size_t n1 = n / 2;
      const FLT *ap = &an[i0 + n1];
      const FLT d0 = dn[n1];
      size_t i;
      if (strides == 1)
      {
        FLT *vp = &output_array[i0];
        const FLT *ia = &input[i0 + n];
        if (n < h)
        {
          for (i = 0; i < h; i++)
          {
            (*vp++) -= d0 * (*ia++) * (*ap++);
          }
        }
        else
        {
          for (i = 0; i < 2 * h - n; i++)
          {
            (*vp++) -= d0 * (*ia++) * (*ap++);
          }
        }
      }
      else
      {
        FLT *vp = &output_array[i0 * strides];
        const FLT *ia = &input[i0 + n];
        if (n < h)
        {
          for (i = 0; i < h; i++)
          {
            (*vp) -= d0 * (*ia++) * (*ap++);
            vp += strides;
          }
        }
        else
        {
          for (i = 0; i < 2 * h - n; i++)
          {
            (*vp) -= d0 * (*ia++) * (*ap++);
            vp += strides;
          }
        }
      }
      flops += i * 3;
    }
  }

  // Last block
  size_t i0 = (nL - 1) * h;
  for (size_t n = 1; n < N - i0; n++)
  {
    const FLT *ap = &an[n + i0];
    const FLT *ia = &input[i0 + 2 * n];
    FLT *op = &output_array[i0 * strides];
    if (strides == 1)
    {
      for (size_t i = i0; i < N - 2 * n; i++)
      {
        (*op++) -= dn[n] * (*ap++) * (*ia++);
      }
    }
    else
    {
      for (size_t i = i0; i < N - 2 * n; i++)
      {
        (*op) -= dn[n] * (*ap++) * (*ia++);
        op += strides;
      }
    }
    flops += (N - (2 * n + i0)) * 3;
  }

  // Multiply result by (x+1/2)
  op = &output_array[0];
  if (strides == 1)
  {
    for (size_t i = 0; i < N; i++)
    {
      (*op++) *= (i + 0.5);
    }
  }
  else
  {
    for (size_t i = 0; i < N; i++)
    {
      (*op) *= (i + 0.5);
      op += strides;
    }
  }
  return flops;
}

void X(matvectri)(const FLT *A, const FLT *x, FLT *b, FLT *w, const size_t m, const bool upper)
{
  // compact triangular matrix A
  size_t i, j;
  if (upper == false)
  {
    FLT *zp = &w[0];
    FLT *zm = &w[m];
    const FLT *xp = &x[m];

    if (m % 2 == 0)
    {
      for (i = 0; i < m; i = i + 2)
      {
        zp[i] = x[i] + xp[i];
        zm[i] = x[i] - xp[i];
        zp[i + 1] = x[i + 1] - xp[i + 1];
        zm[i + 1] = x[i + 1] + xp[i + 1];
      }
      const FLT *a0 = &A[0];
      for (i = 0; i < m; i = i + 2)
      {
        const FLT *z0 = &zp[0];
        const FLT *z1 = &zm[0];
        const FLT *a1 = a0 + i + 1;
        FLT s0 = (*a0++) * (*z0++);
        FLT s1 = (*a1++) * (*z1++);
        for (size_t j = 1; j < i + 1; j++)
        {
          s0 += (*a0++) * (*z0++);
          s1 += (*a1++) * (*z1++);
        }
        s1 += (*a1++) * (*z1);
        a0 = a1;
        b[i] = s0;
        b[i + 1] = s1;
      }
    }
    else
    {
      for (i = 0; i < m - 2; i = i + 2)
      {
        zp[i] = x[i] + xp[i];
        zm[i] = x[i] - xp[i];
        zp[i + 1] = x[i + 1] - xp[i + 1];
        zm[i + 1] = x[i + 1] + xp[i + 1];
      }
      zp[i] = x[i] + xp[i];
      zm[i] = x[i] - xp[i];

      const FLT *a0 = &A[0];
      for (i = 0; i < m - 2; i = i + 2)
      {
        const FLT *a1 = a0 + i + 1;
        FLT s0 = (*a0++) * zp[0];
        FLT s1 = (*a1++) * zm[0];
        for (j = 1; j < i + 1; j++)
        {
          s0 += (*a0++) * zp[j];
          s1 += (*a1++) * zm[j];
        }
        s1 += (*a1++) * zm[j];
        a0 = a1;
        b[i] = s0;
        b[i + 1] = s1;
      }
      FLT s0 = (*a0++) * zp[0];
      for (j = 1; j < i + 1; j++)
      {
        s0 += (*a0++) * zp[j];
      }
      b[i] = s0;
    }
  }
  else
  {
    FLT *bp = &b[m];
    const FLT *ap = &A[0];
    for (i = 0; i < m; i++)
    {
      FLT se = 0.0;
      FLT so = 0.0;
      for (j = i; j < m - 1; j = j + 2)
      {
        se += (*ap++) * x[j];
        so += (*ap++) * x[j + 1];
      }
      if ((i + m) % 2 == 1) se += (*ap++) * x[j];
      (*b++) += se + so;
      (*bp++) += se - so;
    }
  }
}

void X(vandermonde)(FLT *T, const size_t h, const size_t N)
{
  FLT *x = (FLT *)X(fftw_malloc)(2 * h * sizeof(FLT));
  FLT *x2 = (FLT *)X(fftw_malloc)(2 * h * sizeof(FLT));
  FLT *Tm = (FLT *)X(fftw_malloc)(2 * h * N * sizeof(FLT));

  for (size_t i = 0; i < 2 * h; i++)
  {
    x[i] = -1 + (FLT)(i) / ((FLT)h);
    x2[i] = 2 * x[i];
    Tm[i * N] = 1;
    Tm[i * N + 1] = x[i];
  }

  for (size_t i = 0; i < 2 * h; i++)
  {
    for (size_t j = 2; j < N; j++)
    {
      Tm[i * N + j] = Tm[i * N + j - 1] * x2[i] - Tm[i * N + j - 2];
    }
  }

  for (size_t i = 0; i < h; i++)
  {
    for (size_t j = 0; j < N; j++)
    {
      T[i * N + j] = Tm[2 * i * N + j];             // even
      T[(h + i) * N + j] = Tm[(2 * i + 1) * N + j]; // odd
    }
  }
  fftw_free(x);
  fftw_free(x2);
  fftw_free(Tm);
}

void X(free_direct)(X(direct_plan) * plan)
{
  if (plan->a != NULL)
  {
    fftw_free(plan->a);
    plan->a = NULL;
  }
  if (plan->an != NULL)
  {
    fftw_free(plan->an);
    plan->an = NULL;
  }
  if (plan->dn != NULL)
  {
    fftw_free(plan->dn);
    plan->dn = NULL;
  }
  free(plan);
  plan = NULL;
}

void X(free_fmm_2d)(X(fmm_plan_2d) * plan)
{
  if (plan->fmmplan0 == plan->fmmplan1)
  {
    X(free_fmm)(plan->fmmplan0);
    plan->fmmplan1 = NULL;
  }
  else if (plan->fmmplan0 != NULL)
  {
    X(free_fmm)(plan->fmmplan0);
  }
  else if (plan->fmmplan1 != NULL)
  {
    X(free_fmm)(plan->fmmplan1);
  }
  free(plan);
  plan = NULL;
}

void X(free_fmm)(X(fmm_plan) * plan)
{
  if (plan->A[0] != NULL)
  {
    fftw_free(plan->A[0]);
    plan->A[0] = NULL;
  }
  if (plan->A[1] != NULL)
  {
    fftw_free(plan->A[1]);
    plan->A[1] = NULL;
  }
  if (plan->A != NULL)
  {
    free(plan->A);
    plan->A = NULL;
  }
  if (plan->T != NULL)
  {
    fftw_free(plan->T);
    plan->T = NULL;
  }
  if (plan->M != 18)
  {
    if (plan->B != NULL)
    {
      fftw_free(plan->B);
      plan->B = NULL;
    }
    if (plan->BT != NULL)
    {
      fftw_free(plan->BT);
      plan->BT = NULL;
    }
  }
  if (plan->ia != NULL)
  {
    fftw_free(plan->ia);
    plan->ia = NULL;
  }
  if (plan->oa != NULL)
  {
    fftw_free(plan->ia);
    plan->ia = NULL;
  }
  if (plan->work != NULL)
  {
    fftw_free(plan->work);
    plan->work = NULL;
  }
  if (plan->wk != NULL)
  {
    fftw_free(plan->wk[0]);
    for (size_t level = 0; level < plan->L; level++)
      plan->wk[level] = NULL;
    fftw_free(plan->wk);
  }
  if (plan->ck != NULL)
  {
    fftw_free(plan->ck[0]);
    for (size_t level = 0; level < plan->L; level++)
      plan->ck[level] = NULL;
    fftw_free(plan->ck);
  }
  if (plan->dplan != NULL)
  {
    X(free_direct)(plan->dplan);
  }
  free(plan);
  plan = NULL;
}

X(direct_plan) * X(create_direct)(size_t N, size_t direction)
{
  X(direct_plan) *dplan = (X(direct_plan) *)malloc(sizeof(X(direct_plan)));
  FLT *a = (FLT *)X(fftw_malloc)(N * sizeof(FLT));
  dplan->a = a;
  dplan->an = NULL;
  dplan->dn = NULL;

  size_t N0 = 256 < N ? 256 : N;
  for (size_t i = 0; i < N0; i++) // Integer, precomputed values
    a[i] = X(LambdaI)(i);
  size_t DN = 8;
  for (size_t i = N0; i < N; i += DN)
  {
    a[i] = X(LambdaI)(i);
    size_t J0 = i + DN < N ? i + DN : N;
    for (size_t j = i + 1; j < J0; j++)
    {
      // a[j] = a[j - 1] * (1 - 0.5 / j);
      // a[j] = ((j << 1) -1) * a[j - 1] / (j << 1) ;
      a[j] = a[j - 1] - 0.5 * a[j - 1] / j;
    }
  }

  if ((direction == C2L) | (direction == BOTH))
  {
    FLT *dn = (FLT *)X(fftw_malloc)((N + 1) / 2 * sizeof(FLT));
    FLT *an = (FLT *)X(fftw_malloc)(N * sizeof(FLT));
    dn[0] = 0;
    an[0] = M_2_SQRTPI;
    // Using Lambda(i-0.5) = 1/(i*Lambda(i))
    for (size_t i = 1; i < N; i++)
    {
      an[i] = 1 / (a[i] * (2 * i * i + i));
    }
    for (size_t i = 1; i < (N + 1) / 2; i++)
      dn[i] = a[i - 1] / (2 * i);

    dplan->an = an;
    dplan->dn = dn;
  }
  dplan->direction = direction;
  dplan->N = N;

  return dplan;
}

size_t X(direct)(const FLT *u, FLT *b, X(direct_plan) * dplan, size_t direction, size_t strides)
{
  size_t flops = 0;
  const FLT sqrt_pi = 1.77245385090551602e0;
  const size_t N = dplan->N;
  for (size_t i = 0; i < N; i++)
    b[i] = 0.0;
  flops += N;
  if (direction == L2C)
  {
    const FLT *a = dplan->a;
    for (size_t n = 0; n < N; n = n + 2)
    {
      const FLT *ap = &a[n / 2];
      const FLT *cp = &u[n];
      const FLT a0 = ap[0] * M_2_PI;
      for (size_t i = 0; i < N - n; i++)
      {
        b[i * strides] += a0 * ap[i] * cp[i];
      }
      flops += 3 * (N - n);
    }
    b[0] /= 2;
    flops += N;
  }
  else
  {
    FLT *vn = (FLT *)X(fftw_malloc)(N * sizeof(FLT));
    const FLT *an = dplan->an;
    const FLT *dn = dplan->dn;
    vn[0] = u[0];
    for (size_t i = 1; i < N; i++)
      vn[i] = u[i * strides] * i;

    for (size_t n = 0; n < N; n++)
      b[n * strides] = sqrt_pi * vn[n] * an[n];

    for (size_t n = 2; n < N; n = n + 2)
    {
      const FLT *ap = &an[n / 2];
      const FLT *vp = &vn[n];
      for (size_t i = 0; i < N - n; i++)
        b[i * strides] -= dn[n / 2] * ap[i] * vp[i];
      flops += 3 * (N - n);
    }
    for (size_t i = 0; i < N; i++)
      b[i * strides] *= (i + 0.5);
    flops += N;
    fftw_free(vn);
  }

  return flops;
}

X(fmm_plan) * X(create_fmm)(const size_t N, const size_t maxs, const size_t M,
                            const size_t direction, const size_t lagrange, const size_t v)
{
  X(fmm_plan) *fmmplan = (X(fmm_plan) *)malloc(sizeof(X(fmm_plan)));
  X(fftw_plan) plan1d, plan;
  uint64_t t1 = tic;
  size_t Nn;
  size_t s;
  size_t ij[2];
  size_t directions[2];
  size_t num_directions = 2;
  switch (direction)
  {
  case L2C:
    directions[0] = 0;
    num_directions = 1;
    break;
  case C2L:
    directions[0] = 1;
    num_directions = 1;
    break;
  default:
    directions[0] = 0;
    directions[1] = 1;
    break;
  }
  FLT **A = (FLT **)calloc(2, sizeof(FLT *));
  fmmplan->A = A;
  fmmplan->T = NULL;
  fmmplan->B = NULL;
  fmmplan->BT = NULL;
  fmmplan->ia = NULL;
  fmmplan->oa = NULL;
  fmmplan->work = NULL;
  fmmplan->wk = NULL;
  fmmplan->ck = NULL;
  fmmplan->dplan = NULL;
  fmmplan->lagrange = lagrange;

  int L = ceil(log2((FLT)N / (FLT)maxs)) - 2;
  if (L < 1)
  {
    if (v > 1) printf("Levels < 1. Using only direct method\n");
    fmmplan->dplan = X(create_direct)(N, direction);
    fmmplan->Nn = N;
    fmmplan->L = 0;
    return fmmplan;
  }

  s = ceil((FLT)N / (FLT)pow(2, L + 2));
  Nn = s * pow(2, L + 2);

  fmmplan->dplan = X(create_direct)(N, direction);
  fmmplan->M = M;
  fmmplan->L = L;
  fmmplan->N = N;
  fmmplan->Nn = Nn;
  fmmplan->s = s;
  if (v > 1)
  {
    printf("N %lu\n", N);
    printf("Num levels %d\n", L);
    printf("Num submatrices %lu\n", get_total_number_of_submatrices(L));
    printf("Num blocks %lu\n", get_total_number_of_blocks(L));
    printf("Given max s %lu \n", maxs);
    printf("Computed s %lu \n", s);
    printf("Computed N %lu\n", Nn);
    printf("Lagrange %lu\n", lagrange);
  }

  FLT *fun = (FLT *)X(fftw_malloc)(M * M * sizeof(FLT));
  FLT *fun_hat = (FLT *)X(fftw_malloc)(M * M * sizeof(FLT));
  bool use_FFTW = true;

#if defined(FT_USE_DOUBLE)
  use_FFTW = (M != 18 && lagrange == 0);
#endif

  if (use_FFTW)
  {
    if (v > 1) printf("using FFTW for planning\n");
    plan1d = X(fftw_plan_r2r_1d)(M, fun, fun_hat);
    plan = X(fftw_plan_r2r_2d)(M, fun, fun_hat);
  }

  const size_t MM = M * M;
  if (direction == BOTH)
  {
    A[0] = (FLT *)X(fftw_malloc)(get_total_number_of_submatrices(L) * MM * sizeof(FLT));
    A[1] = (FLT *)X(fftw_malloc)(get_total_number_of_submatrices(L) * MM * sizeof(FLT));
  }
  else
  {
    A[direction] = (FLT *)X(fftw_malloc)(get_total_number_of_submatrices(L) * MM * sizeof(FLT));
  }

  FLT *xj = (FLT *)X(fftw_malloc)(M * sizeof(FLT));
  FLT *xjh = (FLT *)X(fftw_malloc)(M * sizeof(FLT));
  for (size_t i = 0; i < M; i++)
  {
    xj[i] = cos((i + 0.5) * M_PI / M);
  }

  FLT *fx0 = (FLT *)X(fftw_malloc)(2 * MM * sizeof(FLT));
  FLT *fx1 = (FLT *)X(fftw_malloc)(MM * sizeof(FLT));
  FLT *lx1 = (FLT *)X(fftw_malloc)(MM * sizeof(FLT));

  size_t kk = 0;
  for (size_t level = 0; level < L; level++)
  {
    size_t h = s * get_h(level, L);
    for (size_t k = 0; k < M; k++)
      xjh[k] = xj[k] * h;
    for (size_t block = 0; block < get_number_of_blocks(level); block++)
    {
      get_ij(ij, level, block, s, L);
      for (size_t q = 0; q < 2; q++)
      {
        size_t y0 = 2 * (ij[1] + q * h) + h;
        for (size_t p = 0; p < q + 1; p++)
        {
          size_t x0 = 2 * (ij[0] + p * h) + h;
          for (size_t di = 0; di < num_directions; di++)
          {
            const size_t dir = directions[di];
            //  ff is input to the DCT
            FLT *ff = (lagrange == 0) ? &fun[0] : &A[dir][kk * MM];
            FLT *f00 = &fx0[q * MM];
            FLT *fpq = &fx0[(q - p) * MM];

            for (size_t i = 0; i < M; i++)
            {
              FLT x = x0 + xjh[i];
              for (size_t j = 0; j < M; j++)
              {
                FLT y = y0 + xjh[j];
                size_t ix = i * M + j;
                size_t xi = j * M + i;
                if (di == 0 && block == 0 && p == 0)
                {
                  if (j < M - i)
                  { // Persymmetric Lambda((y-x)/2)
                    FLT m0 = X(Lambda)((y - x) / 2);
                    // FLT m0 = LambdaE((y - x) / 2);
                    f00[ix] = m0;
                    f00[MM - xi - 1] = m0;
                  }
                }
                if (di == 0)
                {
                  if (j >= i)
                  { // Symmetric Lambda((x+y)/2)
                    FLT m1 = X(Lambda)((x + y) / 2);
                    // FLT m1 = LambdaE((x + y) / 2);
                    fx1[ix] = m1;
                    fx1[xi] = m1;
                  }
                }
                if (dir == L2C)
                {
                  (*ff++) = fpq[ix] * fx1[ix];
                }
                else
                {
                  (*ff++) = 2 * (fpq[ix] / (fx1[ix] * (x + y) * (x + y + 1) * (x - y + 1)));
                }
              }
            }

            if (lagrange == 0)
            {
#if defined(FT_USE_DOUBLE)
              if (use_FFTW)
#endif
              {
                X(fftw_execute)(plan);
                for (size_t i = 0; i < M; i++)
                {
                  for (size_t j = 0; j < M; j++)
                  {
                    fun_hat[i * M + j] *= (1. / (M * M));
                  }
                }
                for (size_t i = 0; i < M; i++)
                {
                  fun_hat[i] /= 2;
                  fun_hat[i * M] /= 2;
                }
                memcpy(&A[dir][kk * MM], &fun_hat[0], MM * sizeof(FLT));
              }
#if defined(FT_USE_DOUBLE)
              else
              {
                dct2(&fun[0], &A[dir][kk * MM]);
              }
#endif
            }
          }
          kk += 1;
        }
      }
    }
  }

  FLT *wj = NULL;
  if (lagrange == 1)
  {
    wj = (FLT *)X(fftw_malloc)(M * sizeof(FLT));
    for (size_t j = 0; j < M; j++)
    {
      int sign = (j % 2 == 0) ? 1 : -1;
      wj[j] = sign * sin((j + 0.5) * M_PI / M);
    }
  }

  FLT *T = (FLT *)X(fftw_malloc)(2 * s * M * sizeof(FLT));
  if (lagrange == 0)
    X(vandermonde)(T, s, M);
  else
  {
    FLT *xh = (FLT *)X(fftw_malloc)(2 * s * sizeof(FLT));
    for (size_t i = 0; i < 2 * s; i++)
    {
      xh[i] = -1 + (FLT)(i) / ((FLT)s);
    }
    for (size_t i = 0; i < s; i++)
    {
      FLT sume = 0.0;
      FLT sumo = 0.0;
      for (size_t j = 0; j < M; j++)
      {
        FLT se = wj[j] / (xh[2 * i] - xj[j]);
        FLT so = wj[j] / (xh[2 * i + 1] - xj[j]);
        T[i * M + j] = se;       // even
        T[(s + i) * M + j] = so; // odd
        sume += se;
        sumo += so;
      }
      for (size_t j = 0; j < M; j++)
      {
        T[i * M + j] /= sume;
        T[(s + i) * M + j] /= sumo;
      }
    }
    fftw_free(xh);
  }
  fmmplan->T = T;

  FLT *B = NULL;
  FLT *BT = NULL;
#if defined(FT_USE_DOUBLE)
  if (!use_FFTW && lagrange == 0)
  {
    if (v > 1) printf("Using exact binomial matrix\n");
    B = (FLT *)&BMe[0];
    BT = (FLT *)&BMeT[0];
  }
  else
#endif
  {
    if (lagrange == 0)
    {
      B = (FLT *)X(fftw_malloc)((MM + M) / 2 * sizeof(FLT));
      BT = (FLT *)X(fftw_malloc)((MM + M) / 2 * sizeof(FLT));
      FLT *Ba = (FLT *)X(fftw_malloc)(MM * sizeof(FLT));
      FLT *BTa = (FLT *)X(fftw_malloc)(MM * sizeof(FLT));
      FLT *th = &Ba[0];
      for (size_t k = 0; k < M; k++)
      {
        for (size_t j = 0; j < M; j++)
        {
          fun[j] = cos(k * acos((xj[j] - 1) / 2));
          fun_hat[j] = 0.0;
        }
        X(fftw_execute)(plan1d);
        *th++ = fun_hat[0] / M / 2;
        for (size_t j = 1; j < M; j++)
          *th++ = fun_hat[j] / M;
      }
      // Make transpose
      th = &Ba[0];
      FLT *tht = &BTa[0];
      for (size_t i = 0; i < M; i++)
      {
        for (size_t j = 0; j < M; j++)
        {
          tht[i * M + j] = th[j * M + i];
        }
      }
      /// Move to compact storage
      th = &B[0];
      for (size_t k = 0; k < M; k++)
      {
        for (size_t j = 0; j < k + 1; j++)
        {
          (*th++) = Ba[k * M + j];
        }
      }
      tht = &BT[0];
      for (size_t k = 0; k < M; k++)
      {
        for (size_t j = k; j < M; j++)
        {
          (*tht++) = BTa[k * M + j];
        }
      }
      fftw_free(Ba);
      fftw_free(BTa);
    }
    else
    {
      B = (FLT *)X(fftw_malloc)(2 * MM * sizeof(FLT));
      FLT *xh = (FLT *)X(fftw_malloc)(2 * M * sizeof(FLT));
      for (size_t i = 0; i < M; i++)
      {
        xh[i] = (xj[i] - 1) / 2;
        xh[M + i] = (xj[i] + 1) / 2;
      }
      for (size_t i = 0; i < M; i++)
      {
        FLT sum0 = 0.0;
        FLT sum1 = 0.0;
        for (size_t j = 0; j < M; j++)
        {
          FLT s0 = wj[j] / (xh[i] - xj[j]);
          FLT s1 = wj[j] / (xh[M + i] - xj[j]);
          B[i * M + j] = s0;
          B[MM + i * M + j] = s1;
          sum0 += s0;
          sum1 += s1;
        }
        for (size_t j = 0; j < M; j++)
        {
          B[i * M + j] /= sum0;
          B[MM + i * M + j] /= sum1;
        }
      }
      fftw_free(xh);
    }
  }

  FLT *ia = (FLT *)X(fftw_malloc)(Nn / 2 * sizeof(FLT));
  FLT *oa = (FLT *)X(fftw_malloc)(Nn / 2 * sizeof(FLT));
  FLT *work = (FLT *)X(fftw_malloc)(2 * M * sizeof(FLT));
  FLT **wk = (FLT **)X(fftw_malloc)(L * sizeof(FLT *));
  FLT **ck = (FLT **)X(fftw_malloc)(L * sizeof(FLT *));
  size_t Nb = get_total_number_of_blocks(L);
  wk[0] = (FLT *)X(fftw_malloc)(Nb * 2 * M * sizeof(FLT));
  ck[0] = (FLT *)X(fftw_malloc)(Nb * 2 * M * sizeof(FLT));
  for (size_t level = 1; level < L; level++)
  {
    size_t b = get_number_of_blocks(level - 1);
    wk[level] = wk[level - 1] + b * 2 * M;
    ck[level] = ck[level - 1] + b * 2 * M;
  }
  fmmplan->ia = ia;
  fmmplan->oa = oa;
  fmmplan->wk = wk;
  fmmplan->ck = ck;
  fmmplan->work = work;
  if (use_FFTW)
  {
    X(fftw_destroy_plan)(plan);
    X(fftw_destroy_plan)(plan1d);
  }

  fftw_free(fun);
  fftw_free(fun_hat);
  fftw_free(fx0);
  fftw_free(fx1);
  fftw_free(lx1);
  fftw_free(xj);
  fftw_free(xjh);
  if (lagrange == 1)
  {
    fftw_free(wj);
  }
  fmmplan->B = B;
  fmmplan->BT = BT;
  uint64_t t2 = tic;
  if (v > 1) printf("Initialization %2.4e s\n", dtics(t1, t2));
  return fmmplan;
}

X(fmm_plan_2d) * X(create_fmm_2d)(size_t N0, size_t N1, int axis, size_t maxs, size_t M,
                                  size_t direction, size_t lagrange, size_t v)
{
  X(fmm_plan_2d) *fmmplan2d = (X(fmm_plan_2d) *)malloc(sizeof(X(fmm_plan_2d)));
  fmmplan2d->fmmplan0 = NULL;
  fmmplan2d->fmmplan1 = NULL;
  if (v > 1)
  {
    printf("crate_fmm_2d\n");
  }
  if (axis == 0)
  {
    fmmplan2d->fmmplan0 = X(create_fmm)(N0, maxs, M, direction, lagrange, v);
  }
  else if (axis == 1)
  {
    fmmplan2d->fmmplan1 = X(create_fmm)(N1, maxs, M, direction, lagrange, v);
  }
  else if (axis == -1)
  {
    fmmplan2d->fmmplan0 = X(create_fmm)(N0, maxs, M, direction, lagrange, v);
    if (N0 == N1)
    {
      fmmplan2d->fmmplan1 = fmmplan2d->fmmplan0;
    }
    else
    {
      fmmplan2d->fmmplan1 = X(create_fmm)(N1, maxs, M, direction, lagrange, v);
    }
  }
  fmmplan2d->N0 = N0;
  fmmplan2d->N1 = N1;
  fmmplan2d->axis = axis;
  return fmmplan2d;
}

size_t X(execute2D)(const FLT *input_array, FLT *output_array, X(fmm_plan_2d) * fmmplan2d,
                    size_t direction)
{
  size_t flops = 0;
  if (fmmplan2d->axis == 0)
  {
    for (size_t i = 0; i < fmmplan2d->N1; i++)
    {
      flops += X(execute)(&input_array[i], &output_array[i], fmmplan2d->fmmplan0, direction,
                          fmmplan2d->N1);
    }
  }
  else if (fmmplan2d->axis == 1)
  {
    for (size_t i = 0; i < fmmplan2d->N0; i++)
    {
      size_t N1 = fmmplan2d->N1;
      flops += X(execute)(&input_array[i * N1], &output_array[i * N1], fmmplan2d->fmmplan1,
                          direction, 1);
    }
  }
  else if (fmmplan2d->axis == -1)
  {
    FLT *out = (FLT *)calloc(fmmplan2d->N0 * fmmplan2d->N1, sizeof(FLT));
    for (size_t i = 0; i < fmmplan2d->N1; i++)
    {
      flops += X(execute)(&input_array[i], &out[i], fmmplan2d->fmmplan0, direction, fmmplan2d->N1);
    }

    for (size_t i = 0; i < fmmplan2d->N0; i++)
    {
      size_t N1 = fmmplan2d->N1;
      flops += X(execute)(&out[i * N1], &output_array[i * N1], fmmplan2d->fmmplan1, direction, 1);
    }
    fftw_free(out);
  }
  return flops;
}

size_t X(execute)(const FLT *input_array, FLT *output_array, X(fmm_plan) * fmmplan,
                  size_t direction, const size_t stride)
{
  size_t Nn = fmmplan->Nn;
  size_t N = fmmplan->N;
  size_t L = fmmplan->L;
  size_t s = fmmplan->s;
  size_t M = fmmplan->M;
  FLT *T = fmmplan->T;
  FLT *B = fmmplan->B;
  FLT *BT = fmmplan->BT;
  FLT *A = fmmplan->A[direction];
  size_t flops = 0;
  size_t lagrange = fmmplan->lagrange;
  assert((direction == C2L) | (direction == L2C));

  if (T == NULL)
  {
    flops = X(direct)(input_array, output_array, fmmplan->dplan, direction, stride);
    return flops;
  }

  FLT *ia = fmmplan->ia;
  FLT *oa = fmmplan->oa;
  FLT **wk = fmmplan->wk;
  FLT **ck = fmmplan->ck;
  FLT *input = NULL;
  if (direction == C2L)
  { // Need to modify input array, so make copy
    input = (FLT *)X(fftw_malloc)(N * sizeof(FLT));
    input[0] = input_array[0];
    input[1] = input_array[stride];
    FLT *w0 = &input[2];
    size_t ii = 2 * stride;
    for (size_t i = 2; i < N; i++)
    {
      (*w0++) = input_array[ii] * i;
      ii += stride;
    }
  }


  for (size_t odd = 0; odd < 2; odd++)
  {
    for (size_t i = 0; i < Nn / 2; i++)
    {
      oa[i] = 0.0;
    }

    memset(&wk[0][0], 0, 2 * M * get_total_number_of_blocks(L) * sizeof(FLT));
    memset(&ck[0][0], 0, 2 * M * get_total_number_of_blocks(L) * sizeof(FLT));

    const FLT *ap;
    switch (direction)
    {
    case L2C:
      ap = &input_array[odd * stride];
      break;

    case C2L:
      ap = &input[odd];
      break;
    }

    size_t rest = N % 2;
    FLT *iap = &ia[0];
    if (stride == 1 || direction == C2L)
    {
      for (size_t i = 0; i < N / 2 + rest * (1 - odd); i++)
      {
        *iap++ = *ap;
        ap += 2;
      }
    }
    else
    {
      for (size_t i = 0; i < N / 2 + rest * (1 - odd); i++)
      {
        *iap++ = *ap;
        ap += 2 * stride;
      }
    }

    for (size_t i = N / 2 + rest * (1 - odd); i < Nn / 2; i++)
    {
      *iap++ = 0;
    }

    const size_t MM = M * M;
    const size_t K = get_number_of_blocks(L - 1) * 2;

    X(rm_gemm)('N', K, M, s, 1.0, &ia[2 * s], s, &T[odd * s * M], M, 0, wk[L - 1], M);
    flops += 2 * K * s * M;
    for (size_t level = L; level-- > 1;)
    {
      FLT *w1 = wk[level - 1];
      for (size_t block = 1; block < get_number_of_blocks(level); block++)
      {
        size_t Nd = block * 2 * M;
        FLT *wq = &wk[level][Nd];
        int b0 = (block - 1) / 2;
        int q0 = (block - 1) % 2;
        if (lagrange == 0)
        {
          X(matvectri)(&B[0], wq, &w1[(b0 * 2 + q0) * M], fmmplan->work, M, false);
          flops += MM; //+2*M;
        }
        else
        {
          X(rm_gemv)('T', M, M, 1, &B[0], M, &wq[0], 0, &w1[(b0 * 2 + q0) * M]);
          X(rm_gemv)('T', M, M, 1, &B[MM], M, &wq[M], 1, &w1[(b0 * 2 + q0) * M]);
          flops += 4 * MM;
        }
      }
    }

    size_t ik = 0;
    for (size_t level = 0; level < L; level++)
    {
      for (size_t block = 0; block < get_number_of_blocks(level); block++)
      {
        size_t Nd = block * 2 * M;
        FLT *cp = &ck[level][Nd];
        FLT *wq = &wk[level][Nd];
        X(rm_gemv)('N', M, M, 1, &A[ik * MM], M, wq, 0, cp);
        X(rm_gemv)('N', M, M, 1, &A[(ik + 1) * MM], M, &wq[M], 1, cp);
        X(rm_gemv)('N', M, M, 1, &A[(ik + 2) * MM], M, &wq[M], 0, &cp[M]);
        flops += 6 * MM;
        ik += 3;
      }
    }

    for (size_t level = 0; level < L - 1; level++)
    {
      FLT *c0 = ck[level];
      FLT *c1 = ck[level + 1];
      for (size_t block = 0; block < get_number_of_blocks(level + 1) - 1; block++)
      {
        if (lagrange == 0)
        {
          X(matvectri)(&BT[0], &c0[block * M], &c1[block * 2 * M], NULL, M, true);
          flops += MM;
        }
        else
        {
          X(rm_gemv)('N', M, M, 1, &B[0], M, &c0[block * M], 1, &c1[block * 2 * M]);
          X(rm_gemv)('N', M, M, 1, &B[MM], M, &c0[block * M], 1, &c1[block * 2 * M + M]);
          flops += 4 * MM;
        }
      }
    }
    X(rm_gemm)('T', K, s, M, 1.0, ck[L - 1], M, &T[odd * s * M], M, 0, &oa[0], s);

    flops += 2 * K * s * M;
    FLT *oaa = &output_array[odd * stride];
    FLT *oap = &oa[0];
    if (stride == 1)
    {
      for (size_t i = 0; i < N; i = i + 2)
      {
        *oaa += (*oap++);
        oaa += 2;
      }
    }
    else
    {
      const size_t s2 = 2 * stride;
      for (size_t i = 0; i < N / 2 + rest * (1 - odd); i++)
      {
        *oaa += (*oap++);
        oaa += s2;
      }
    }
    // flops += 3*N/2;
  }

  switch (direction)
  {
  case L2C:
    flops += X(directM)(input_array, output_array, fmmplan, stride);
    break;

  case C2L:
    flops += X(directL)(input, output_array, fmmplan, stride);
    break;
  }

  if (input != NULL) fftw_free(input);
  return flops;
}
