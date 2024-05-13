
typedef struct {
  size_t direction;
  size_t N;
  FLT *a;
  FLT *dn;
  FLT *an;
} X(direct_plan);

typedef struct {
  size_t direction;
  size_t M;
  size_t N;
  size_t Nn;
  size_t L;
  size_t s;
  size_t lagrange;
  FLT **A;
  FLT *T;
  FLT *B;
  FLT *BT;
  FLT *ia;
  FLT *oa;
  FLT *work;
  FLT **wk;
  FLT **ck;
  X(direct_plan) * dplan;
} X(fmm_plan);

typedef struct {
  X(fmm_plan) * fmmplan0;
  X(fmm_plan) * fmmplan1;
  size_t N0;
  size_t N1;
  int axis;
} X(fmm_plan_2d);

void X(rm_gemv)(char TRANS, int m, int n, FLT alpha, FLT *A, int LDA, FLT *x,
                FLT beta, FLT *y);
void X(rm_gemm)(char TRANS, int m, int n, int p, FLT alpha, FLT *A, int LDA,
                FLT *B, int LDB, FLT beta, FLT *C, int LDC);
void X(free_fmm)(X(fmm_plan) * plan);
void X(free_fmm_2d)(X(fmm_plan_2d) * plan);
void X(free_direct)(X(direct_plan) * dplan);
X(fmm_plan_2d) * X(create_fmm_2d)(size_t N0, size_t N1, int axis, size_t maxs,
                                  size_t M, size_t direction,
                                  const size_t lagrange, size_t v);
X(fmm_plan) * X(create_fmm)(const size_t N, const size_t maxs, const size_t M,
                            const size_t direction, const size_t lagrange,
                            const size_t v);
X(direct_plan) * X(create_direct)(size_t N, size_t direction);
size_t X(execute2D)(const FLT *input_array, FLT *output_array,
                    X(fmm_plan_2d) * fmmplan2d, size_t direction);
size_t X(execute)(const FLT *input_array, FLT *output_array,
                  X(fmm_plan) * fmmplan, size_t direction, const size_t stride);
size_t X(direct)(const FLT *u, FLT *b, X(direct_plan) * dplan, size_t direction,
                 size_t strides);
FLT X(Lambda)(const FLT z);
