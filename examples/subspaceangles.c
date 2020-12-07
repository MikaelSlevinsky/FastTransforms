#include <fasttransforms.h>
#include <ftutilities.h>

#define VtV(i, j) VtV[(i)+(j)*N]

/*!
  \example subspaceangles.c
  This example considers the angles between neighbouring Laguerre polynomials with a perturbed measure:
  \f[
  \cos\theta_n = \dfrac{\langle L_n, L_{n+k}\rangle}{\|L_n|_2 \|L_{n+k}\|_2},\quad{\rm for}\quad 0\le n < N-k,
  \f]
  where the inner product is defined by \f$\langle f, g\rangle = \int_0^\infty f(x) g(x) x^\beta e^{-x}{\rm\,d}x\f$.
  We do so by connecting Laguerre polynomials to the normalized generalized Laguerre polynomials associated with the perturbed measure. It follows by the inner product of the connection coefficients that:
  \f[
  \cos\theta_n = \dfrac{(V^\top V)_{n, n+k}}{\sqrt{(V^\top V)_{n, n}(V^\top V)_{n+k, n+k}}}.
  \f]
*/
int main(void) {
    printf("This example considers the angles between neighbouring Laguerre\n");
    printf("polynomials with a perturbed measure:\n");
    printf("\n");
    printf("\t"MAGENTA("cos θₙ = ⟨Lₙ, Lₙ₊ₖ⟩ / (‖Lₙ‖₂ ‖Lₙ₊ₖ‖₂)")", for "MAGENTA("0 ≤ n < N-k")",\n");
    printf("\n");
    printf("where the inner product is defined by "MAGENTA("⟨f, g⟩ = ∫ f(x) g(x) x^β e^{-x} dx")".\n");
    printf("This is accomplished by using \n");
    printf("\t"CYAN("ft_plan_laguerre_to_laguerre")" and "CYAN("ft_bfmm")".\n");

    char * FMT = "%17.16e";
    int k = 1, N = 11;
    double alpha = 0.0, beta = 0.125, theta[N-k], VtV[N*N];

    printf("\n\n"MAGENTA("k = %i")", "MAGENTA("N = %i")", and "MAGENTA("β = %1.3f")".\n\n", k, N, beta);

    ft_tb_eigen_FMM * P = ft_plan_laguerre_to_laguerre(0, 1, N, alpha, beta);

    for (int n = 0; n < N*N; n++)
        VtV[n] = 0.0;
    for (int n = 0; n < N; n++)
        VtV(n, n) = 1.0;
    ft_bfmm('N', P, VtV, N, N);
    ft_bfmm('T', P, VtV, N, N);
    for (int n = 0; n < N-k; n++)
        theta[n] = acos(VtV(n, n+k)/sqrt(VtV(n, n)*VtV(n+k, n+k)));

    printmat(MAGENTA("θ"), FMT, theta, N-k, 1);

    ft_destroy_tb_eigen_FMM(P);

    return 0;
}
