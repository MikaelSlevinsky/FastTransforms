#include <fasttransforms.h>
#include <ftutilities.h>

void oprec(const int n, double * v, const double alpha, const double delta2) {
    if (n > 0)
        v[0] = 1;
    if (n > 1)
        v[1] = (4*alpha+8-(alpha+4)*delta2)/4;
    for (int i = 1; i < n-1; i++)
        v[i+1] = (((2*i+alpha+2)*(2*i+alpha+4)+alpha*(alpha+2))/(2*(i+1)*(2*i+alpha+2))*(2*i+alpha+3)/(i+alpha+3) - delta2/4*(2*i+alpha+3)/(i+1)*(2*i+alpha+4)/(i+alpha+3))*v[i] - (i+alpha+1)/(i+alpha+3)*(2*i+alpha+4)/(2*i+alpha+2)*v[i-1];
}

/*!
  \example nonlocaldiffusion.c
  This example calculates the spectrum of the nonlocal diffusion operator:
  \f[
  \mathcal{L}_\delta u = \int_{\mathbb{S}^2} \rho_\delta(|\mathbf{x}-\mathbf{y}|)\left[u(\mathbf{x}) - u(\mathbf{y})\right] \,\mathrm{d}\Omega(\mathbf{y}),
  \f]
  defined in Eq. (2) of

    R. M. Slevinsky, H. Montanelli, and Q. Du, [A spectral method for nonlocal diffusion operators on the sphere](https://doi.org/10.1016/j.jcp.2018.06.024), *J. Comp. Phys.*, **372**:893--911, 2018.

  In the above, \f$0<\delta<2\f$, \f$-1<\alpha<1\f$, and the kernel:
  \f[
  \rho_\delta(|\mathbf{x}-\mathbf{y}|) = \frac{4(1+\alpha)}{\pi \delta^{2+2\alpha}} \frac{\chi_{[0,\delta]}(|\mathbf{x}-\mathbf{y}|)}{|\mathbf{x}-\mathbf{y}|^{2-2\alpha}},
  \f]
  where \f$\chi_I(\cdot)\f$ is the indicator function on the set \f$I\f$.

  This nonlocal operator is diagonalized by spherical harmonics:
  \f[
  \mathcal{L}_\delta Y_\ell^m(\mathbf{x}) = \lambda_\ell(\alpha, \delta) Y_\ell^m(\mathbf{x}),
  \f]
  and its eigenfunctions are given by the generalized Funk--Hecke formula:
  \f[
  \lambda_\ell(\alpha, \delta) = \frac{(1+\alpha) 2^{2+\alpha}}{\delta^{2+2\alpha}}\int_{1-\delta^2/2}^1 \left[P_\ell(t)-1\right] (1-t)^{\alpha-1} \,\mathrm{d} t.
  \f]
  In the paper, the authors use Clenshaw--Curtis quadrature and asymptotic evaluation of Legendre polynomials to achieve \f$\mathcal{O}(n^2\log n)\f$ complexity for the evaluation of the first \f$n\f$ eigenvalues. With a change of basis, this complexity can be reduced to \f$\mathcal{O}(n\log n)\f$.

  First, we represent:
  \f[
  P_n(t) - 1 = \sum_{j=0}^{n-1} \left[P_{j+1}(t) - P_j(t)\right] = -\sum_{j=0}^{n-1} (1-t) P_j^{(1,0)}(t).
  \f]
  Then, we represent \f$P_j^{(1,0)}(t)\f$ with Jacobi polynomials \f$P_i^{(\alpha,0)}(t)\f$ and we integrate using [DLMF 18.9.16](https://dlmf.nist.gov/18.9.16):
  \f[
  \int_x^1 P_i^{(\alpha,0)}(t)(1-t)^\alpha\,\mathrm{d}t = \left\{ \begin{array}{cc} \frac{(1-x)^{\alpha+1}}{\alpha+1} & \mathrm{for~}i=0,\\ \frac{1}{2i}(1-x)^{\alpha+1}(1+x)P_{i-1}^{(\alpha+1,1)}(x), & \mathrm{for~}i>0.\end{array}\right.
  \f]
  The code below implements this algorithm, making use of the Jacobi--Jacobi transform \ref ft_plan_jacobi_to_jacobi.
*/
int main(void) {
    printf("This example calculates the spectrum of the nonlocal diffusion\n");
    printf("operator defined in Eq. (2) of\n");
    printf("\t"MAGENTA("R. M. Slevinsky, H. Montanelli, and Q. Du, A spectral method for")"\n");
    printf("\t"MAGENTA("nonlocal diffusion operators on the sphere, J. Comp. Phys.,")"\n");
    printf("\t"MAGENTA("372:893--911, 2018.")"\n");
    printf("Please see "CYAN("https://doi.org/10.1016/j.jcp.2018.06.024")" and\n");
    printf("the online documentation for further details.\n");

    char * FMT = "%17.16e";

    int n = 11;
    double alpha = -0.5, delta = 0.025;
    double delta2 = delta*delta;
    double scl = (1+alpha)*(2-delta2/2);

    printf("\n"MAGENTA("n = %i")", "MAGENTA("α = %1.3f")", and "MAGENTA("δ = %1.3f")".\n", n, alpha, delta);

    double lambda[n];

    if (n > 0)
        lambda[0] = 0;
    if (n > 1)
        lambda[1] = -2;
    oprec(n-2, lambda+2, alpha, delta2);

    for (int i = 2; i < n; i++)
        lambda[i] *= -scl/(i-1);

    ft_tb_eigen_FMM * P = ft_plan_jacobi_to_jacobi(0, 0, n-1, 1, 0, alpha, 0);

    ft_bfmv('T', P, lambda+1);

    for (int i = 2; i < n; i++)
        lambda[i] = lambda[i]+lambda[i-1];

    printf("The spectrum, "MAGENTA("0 ≤ ℓ < %i")", ", n);
    printmat(MAGENTA("λ_ℓ(α,δ)"), FMT, lambda, n, 1);
    printf("\n");

    ft_destroy_tb_eigen_FMM(P);

    return 0;
}
