/*!
 \file fasttransforms.h
 \brief fasttransforms.h is the main header file for FastTransforms.
*/
#ifndef FASTTRANSFORMS_H
#define FASTTRANSFORMS_H

#include <cblas.h>
#include <fftw3.h>

typedef double ft_complex[2];

#ifdef _OPENMP
    #include <omp.h>
    #define FT_GET_THREAD_NUM() omp_get_thread_num()
    #define FT_GET_NUM_THREADS() omp_get_num_threads()
    #define FT_GET_MAX_THREADS() omp_get_max_threads()
    #define FT_SET_NUM_THREADS(x) omp_set_num_threads(x)
#else
    #define FT_GET_THREAD_NUM() 0
    #define FT_GET_NUM_THREADS() 1
    #define FT_GET_MAX_THREADS() 1
    #define FT_SET_NUM_THREADS(x)
#endif

#define FT_CONCAT(prefix, name, suffix) prefix ## name ## suffix

void ft_horner(const int n, const double * c, const int incc, const int m, double * x, double * f);
void ft_hornerf(const int n, const float * c, const int incc, const int m, float * x, float * f);

void ft_clenshaw(const int n, const double * c, const int incc, const int m, double * x, double * f);
void ft_clenshawf(const int n, const float * c, const int incc, const int m, float * x, float * f);

void ft_orthogonal_polynomial_clenshaw(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, double * phi0, double * f);
void ft_orthogonal_polynomial_clenshawf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, float * phi0, float * f);

void ft_eigen_eval(const int n, const double * c, const int incc, const double * A, const double * B, const double * C, const int m, double * x, const int sign, double * f);
void ft_eigen_evalf(const int n, const float * c, const int incc, const float * A, const float * B, const float * C, const int m, float * x, const int sign, float * f);
void ft_eigen_evall(const int n, const long double * c, const int incc, const long double * A, const long double * B, const long double * C, const int m, long double * x, const int sign, long double * f);
#if defined(FT_QUADMATH)
    #include <quadmath.h>
    typedef __float128 quadruple;
    void ft_eigen_evalq(const int n, const quadruple * c, const int incc, const quadruple * A, const quadruple * B, const quadruple * C, const int m, quadruple * x, const int sign, quadruple * f);
#endif

#define FT_SN (1U << 0)
#define FT_CN (1U << 1)
#define FT_DN (1U << 2)

#define FT_MODIFIED_NMAX (1U << 30)

#include "tdc.h"

/*!
  \brief Pre-compute a factorization of the connection coefficients between Legendre and Chebyshev polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Legendre}} P_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x).
  \f]
  `normleg` and `normcheb` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_legendre_to_chebyshevf, \ref ft_plan_legendre_to_chebyshevl, and \ref ft_mpfr_plan_legendre_to_chebyshev.
  \param normleg The normalization of the Legendre polynomials: standard (`0`) or orthonormalized (`1`).
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
*/
ft_tb_eigen_FMM * ft_plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Chebyshev and Legendre polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Legendre}} P_\ell(x).
  \f]
  `normcheb` and `normleg` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_chebyshev_to_legendref, \ref ft_plan_chebyshev_to_legendrel, and \ref ft_mpfr_plan_chebyshev_to_legendre.
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param normleg The normalization of the Legendre polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
*/
ft_tb_eigen_FMM * ft_plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n);
/*!
  \brief Pre-compute a factorization of the connection coefficients between ultraspherical polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{(1)} C_\ell^{(\lambda)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{(2)} C_\ell^{(\mu)}(x).
  \f]
  `norm1` and `norm2` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_ultraspherical_to_ultrasphericalf, \ref ft_plan_ultraspherical_to_ultrasphericall, and \ref ft_mpfr_plan_ultraspherical_to_ultraspherical.
  \param norm1 The normalization of the source ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param norm2 The normalization of the target ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the source ultraspherical polynomials.
  \param mu The parameter \f$\mu\f$ of the target ultraspherical polynomials.
*/
ft_tb_eigen_FMM * ft_plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, const double lambda, const double mu);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Jacobi polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{(1)} P_\ell^{(\alpha,\beta)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{(2)} P_\ell^{(\gamma,\delta)}(x).
  \f]
  `norm1` and `norm2` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_jacobi_to_jacobif, \ref ft_plan_jacobi_to_jacobil, and \ref ft_mpfr_plan_jacobi_to_jacobi.
  \param norm1 The normalization of the source Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param norm2 The normalization of the target Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the source Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the source Jacobi polynomials.
  \param gamma The parameter \f$\gamma\f$ of the target Jacobi polynomials, in the place of \f$\alpha\f$.
  \param delta The parameter \f$\delta\f$ of the target Jacobi polynomials, in the place of \f$\beta\f$.
*/
ft_tb_eigen_FMM * ft_plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const double alpha, const double beta, const double gamma, const double delta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Laguerre polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{(1)} L_\ell^{(\alpha)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{(2)} L_\ell^{(\beta)}(x).
  \f]
  `norm1` and `norm2` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_laguerre_to_laguerref, \ref ft_plan_laguerre_to_laguerrel, and \ref ft_mpfr_plan_laguerre_to_laguerre.
  \param norm1 The normalization of the source Laguerre polynomials: standard (`0`) or orthonormalized (`1`).
  \param norm2 The normalization of the target Laguerre polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the source Laguerre polynomials.
  \param beta The parameter \f$\beta\f$ of the target Laguerre polynomials, in the place of \f$\alpha\f$.
*/
ft_tb_eigen_FMM * ft_plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Jacobi and ultraspherical polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x).
  \f]
  `normjac` and `normultra` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_jacobi_to_ultrasphericalf, \ref ft_plan_jacobi_to_ultrasphericall, and \ref ft_mpfr_plan_jacobi_to_ultraspherical.
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
*/
ft_tb_eigen_FMM * ft_plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, const double alpha, const double beta, const double lambda);
/*!
  \brief Pre-compute a factorization of the connection coefficients between ultraspherical and Jacobi polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x).
  \f]
  `normultra` and `normjac` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_ultraspherical_to_jacobif, \ref ft_plan_ultraspherical_to_jacobil, and \ref ft_mpfr_plan_ultraspherical_to_jacobi.
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
*/
ft_tb_eigen_FMM * ft_plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, const double lambda, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Jacobi and Chebyshev polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x).
  \f]
  `normjac` and `normcheb` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_jacobi_to_chebyshevf, \ref ft_plan_jacobi_to_chebyshevl, and \ref ft_mpfr_plan_jacobi_to_chebyshev.
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
*/
ft_tb_eigen_FMM * ft_plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Chebyshev and Jacobi polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x).
  \f]
  `normcheb` and `normjac` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_chebyshev_to_jacobif, \ref ft_plan_chebyshev_to_jacobil, and \ref ft_mpfr_plan_chebyshev_to_jacobi.
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
*/
ft_tb_eigen_FMM * ft_plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between ultraspherical and Chebyshev polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x).
  \f]
  `normultra` and `normcheb` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_ultraspherical_to_chebyshevf, \ref ft_plan_ultraspherical_to_chebyshevl, and \ref ft_mpfr_plan_ultraspherical_to_chebyshev.
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
*/
ft_tb_eigen_FMM * ft_plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, const double lambda);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Chebyshev and ultraspherical polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x).
  \f]
  `normcheb` and `normultra` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_chebyshev_to_ultrasphericalf, \ref ft_plan_chebyshev_to_ultrasphericall, and \ref ft_mpfr_plan_chebyshev_to_ultraspherical.
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
*/
ft_tb_eigen_FMM * ft_plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, const double lambda);

ft_btb_eigen_FMM * ft_plan_associated_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const int c, const double alpha, const double beta, const double gamma, const double delta);
ft_btb_eigen_FMM * ft_plan_associated_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const int c, const double alpha, const double beta);
ft_btb_eigen_FMM * ft_plan_associated_hermite_to_hermite(const int norm1, const int norm2, const int n, const int c);

/*!
  \brief Pre-compute a factorization of the connection coefficients between modified Jacobi polynomials, orthonormal with respect to a rationally modified weight and orthonormal Jacobi polynomials. The rational function is expressed as a ratio of orthonormal Jacobi polynomial expansions:
  \f[
  r(x) = \frac{u(x)}{v(x)} = \frac{\displaystyle \sum_{k=0}^{n_u-1} u_k \tilde{P}_k^{(\alpha,\beta)}(x)}{\displaystyle \sum_{k=0}^{n_v-1} v_k \tilde{P}_k^{(\alpha,\beta)}(x)}.
  \f]
  If \f$n_v \le 1\f$, then \f$r(x)\f$ is polynomial and the algorithm is direct and faster. See also \ref ft_plan_modified_jacobi_to_jacobif and \ref ft_plan_modified_jacobi_to_jacobil.
  \param n The number of modified polynomials, so that their degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
  \param nu The number of coefficients \f$n_u\f$ in the numerator.
  \param u The array of the \f$n_u\f$ coefficients \f$u_k\f$ of the numerator.
  \param nv The number of coefficients \f$n_v\f$ in the denominator.
  \param v The array of the \f$n_v\f$ coefficients \f$v_k\f$ of the denominator.
  \param verbose If nonzero and \f$r(x)\f$ is rational, print the convergence of the pre-computation to standard output.
*/
ft_modified_plan * ft_plan_modified_jacobi_to_jacobi(const int n, const double alpha, const double beta, const int nu, const double * u, const int nv, const double * v, const int verbose);
/*!
  \brief Pre-compute a factorization of the connection coefficients between modified Laguerre polynomials, orthonormal with respect to a rationally modified weight and orthonormal Laguerre polynomials. The rational function is expressed as a ratio of orthonormal Laguerre polynomial expansions:
  \f[
  r(x) = \frac{u(x)}{v(x)} = \frac{\displaystyle \sum_{k=0}^{n_u-1} u_k \tilde{L}_k^{(\alpha)}(x)}{\displaystyle \sum_{k=0}^{n_v-1} v_k \tilde{L}_k^{(\alpha)}(x)}.
  \f]
  If \f$n_v \le 1\f$, then \f$r(x)\f$ is polynomial and the algorithm is direct and faster. See also \ref ft_plan_modified_laguerre_to_laguerref and \ref ft_plan_modified_laguerre_to_laguerrel.
  \param n The number of modified polynomials, so that their degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Laguerre polynomials.
  \param nu The number of coefficients \f$n_u\f$ in the numerator.
  \param u The array of the \f$n_u\f$ coefficients \f$u_k\f$ of the numerator.
  \param nv The number of coefficients \f$n_v\f$ in the denominator.
  \param v The array of the \f$n_v\f$ coefficients \f$v_k\f$ of the denominator.
  \param verbose If nonzero and \f$r(x)\f$ is rational, print the convergence of the pre-computation to standard output.
*/
ft_modified_plan * ft_plan_modified_laguerre_to_laguerre(const int n, const double alpha, const int nu, const double * u, const int nv, const double * v, const int verbose);
/*!
  \brief Pre-compute a factorization of the connection coefficients between modified Hermite polynomials, orthonormal with respect to a rationally modified weight and orthonormal Hermite polynomials. The rational function is expressed as a ratio of orthonormal Hermite polynomial expansions:
  \f[
  r(x) = \frac{u(x)}{v(x)} = \frac{\displaystyle \sum_{k=0}^{n_u-1} u_k \tilde{H}_k(x)}{\displaystyle \sum_{k=0}^{n_v-1} v_k \tilde{H}_k(x)}.
  \f]
  If \f$n_v \le 1\f$, then \f$r(x)\f$ is polynomial and the algorithm is direct and faster. See also \ref ft_plan_modified_hermite_to_hermitef and \ref ft_plan_modified_hermite_to_hermitel.
  \param n The number of modified polynomials, so that their degree is at most \f$n-1\f$.
  \param nu The number of coefficients \f$n_u\f$ in the numerator.
  \param u The array of the \f$n_u\f$ coefficients \f$u_k\f$ of the numerator.
  \param nv The number of coefficients \f$n_v\f$ in the denominator.
  \param v The array of the \f$n_v\f$ coefficients \f$v_k\f$ of the denominator.
  \param verbose If nonzero and \f$r(x)\f$ is rational, print the convergence of the pre-computation to standard output.
*/
ft_modified_plan * ft_plan_modified_hermite_to_hermite(const int n, const int nu, const double * u, const int nv, const double * v, const int verbose);

/// A single precision version of \ref ft_plan_legendre_to_chebyshev.
ft_tb_eigen_FMMf * ft_plan_legendre_to_chebyshevf(const int normleg, const int normcheb, const int n);
/// A single precision version of \ref ft_plan_chebyshev_to_legendre.
ft_tb_eigen_FMMf * ft_plan_chebyshev_to_legendref(const int normcheb, const int normleg, const int n);
/// A single precision version of \ref ft_plan_ultraspherical_to_ultraspherical.
ft_tb_eigen_FMMf * ft_plan_ultraspherical_to_ultrasphericalf(const int norm1, const int norm2, const int n, const float lambda, const float mu);
/// A single precision version of \ref ft_plan_jacobi_to_jacobi.
ft_tb_eigen_FMMf * ft_plan_jacobi_to_jacobif(const int norm1, const int norm2, const int n, const float alpha, const float beta, const float gamma, const float delta);
/// A single precision version of \ref ft_plan_laguerre_to_laguerre.
ft_tb_eigen_FMMf * ft_plan_laguerre_to_laguerref(const int norm1, const int norm2, const int n, const float alpha, const float beta);
/// A single precision version of \ref ft_plan_jacobi_to_ultraspherical.
ft_tb_eigen_FMMf * ft_plan_jacobi_to_ultrasphericalf(const int normjac, const int normultra, const int n, const float alpha, const float beta, const float lambda);
/// A single precision version of \ref ft_plan_ultraspherical_to_jacobi.
ft_tb_eigen_FMMf * ft_plan_ultraspherical_to_jacobif(const int normultra, const int normjac, const int n, const float lambda, const float alpha, const float beta);
/// A single precision version of \ref ft_plan_jacobi_to_chebyshev.
ft_tb_eigen_FMMf * ft_plan_jacobi_to_chebyshevf(const int normjac, const int normcheb, const int n, const float alpha, const float beta);
/// A single precision version of \ref ft_plan_chebyshev_to_jacobi.
ft_tb_eigen_FMMf * ft_plan_chebyshev_to_jacobif(const int normcheb, const int normjac, const int n, const float alpha, const float beta);
/// A single precision version of \ref ft_plan_ultraspherical_to_chebyshev.
ft_tb_eigen_FMMf * ft_plan_ultraspherical_to_chebyshevf(const int normultra, const int normcheb, const int n, const float lambda);
/// A single precision version of \ref ft_plan_chebyshev_to_ultraspherical.
ft_tb_eigen_FMMf * ft_plan_chebyshev_to_ultrasphericalf(const int normcheb, const int normultra, const int n, const float lambda);

ft_btb_eigen_FMMf * ft_plan_associated_jacobi_to_jacobif(const int norm1, const int norm2, const int n, const int c, const float alpha, const float beta, const float gamma, const float delta);
ft_btb_eigen_FMMf * ft_plan_associated_laguerre_to_laguerref(const int norm1, const int norm2, const int n, const int c, const float alpha, const float beta);
ft_btb_eigen_FMMf * ft_plan_associated_hermite_to_hermitef(const int norm1, const int norm2, const int n, const int c);

/// A single precision version of \ref ft_plan_modified_jacobi_to_jacobi.
ft_modified_planf * ft_plan_modified_jacobi_to_jacobif(const int n, const float alpha, const float beta, const int nu, const float * u, const int nv, const float * v, const int verbose);
/// A single precision version of \ref ft_plan_modified_laguerre_to_laguerre.
ft_modified_planf * ft_plan_modified_laguerre_to_laguerref(const int n, const float alpha, const int nu, const float * u, const int nv, const float * v, const int verbose);
/// A single precision version of \ref ft_plan_modified_hermite_to_hermite.
ft_modified_planf * ft_plan_modified_hermite_to_hermitef(const int n, const int nu, const float * u, const int nv, const float * v, const int verbose);

/// A long double precision version of \ref ft_plan_legendre_to_chebyshev.
ft_tb_eigen_FMMl * ft_plan_legendre_to_chebyshevl(const int normleg, const int normcheb, const int n);
/// A long double precision version of \ref ft_plan_chebyshev_to_legendre.
ft_tb_eigen_FMMl * ft_plan_chebyshev_to_legendrel(const int normcheb, const int normleg, const int n);
/// A long double precision version of \ref ft_plan_ultraspherical_to_ultraspherical.
ft_tb_eigen_FMMl * ft_plan_ultraspherical_to_ultrasphericall(const int norm1, const int norm2, const int n, const long double lambda, const long double mu);
/// A long double precision version of \ref ft_plan_jacobi_to_jacobi.
ft_tb_eigen_FMMl * ft_plan_jacobi_to_jacobil(const int norm1, const int norm2, const int n, const long double alpha, const long double beta, const long double gamma, const long double delta);
/// A long double precision version of \ref ft_plan_laguerre_to_laguerre.
ft_tb_eigen_FMMl * ft_plan_laguerre_to_laguerrel(const int norm1, const int norm2, const int n, const long double alpha, const long double beta);
/// A long double precision version of \ref ft_plan_jacobi_to_ultraspherical.
ft_tb_eigen_FMMl * ft_plan_jacobi_to_ultrasphericall(const int normjac, const int normultra, const int n, const long double alpha, const long double beta, const long double lambda);
/// A long double precision version of \ref ft_plan_ultraspherical_to_jacobi.
ft_tb_eigen_FMMl * ft_plan_ultraspherical_to_jacobil(const int normultra, const int normjac, const int n, const long double lambda, const long double alpha, const long double beta);
/// A long double precision version of \ref ft_plan_jacobi_to_chebyshev.
ft_tb_eigen_FMMl * ft_plan_jacobi_to_chebyshevl(const int normjac, const int normcheb, const int n, const long double alpha, const long double beta);
/// A long double precision version of \ref ft_plan_chebyshev_to_jacobi.
ft_tb_eigen_FMMl * ft_plan_chebyshev_to_jacobil(const int normcheb, const int normjac, const int n, const long double alpha, const long double beta);
/// A long double precision version of \ref ft_plan_ultraspherical_to_chebyshev.
ft_tb_eigen_FMMl * ft_plan_ultraspherical_to_chebyshevl(const int normultra, const int normcheb, const int n, const long double lambda);
/// A long double precision version of \ref ft_plan_chebyshev_to_ultraspherical.
ft_tb_eigen_FMMl * ft_plan_chebyshev_to_ultrasphericall(const int normcheb, const int normultra, const int n, const long double lambda);

ft_btb_eigen_FMMl * ft_plan_associated_jacobi_to_jacobil(const int norm1, const int norm2, const int n, const int c, const long double alpha, const long double beta, const long double gamma, const long double delta);
ft_btb_eigen_FMMl * ft_plan_associated_laguerre_to_laguerrel(const int norm1, const int norm2, const int n, const int c, const long double alpha, const long double beta);
ft_btb_eigen_FMMl * ft_plan_associated_hermite_to_hermitel(const int norm1, const int norm2, const int n, const int c);

/// A long double precision version of \ref ft_plan_modified_jacobi_to_jacobi.
ft_modified_planl * ft_plan_modified_jacobi_to_jacobil(const int n, const long double alpha, const long double beta, const int nu, const long double * u, const int nv, const long double * v, const int verbose);
/// A long double precision version of \ref ft_plan_modified_laguerre_to_laguerre.
ft_modified_planl * ft_plan_modified_laguerre_to_laguerrel(const int n, const long double alpha, const int nu, const long double * u, const int nv, const long double * v, const int verbose);
/// A long double precision version of \ref ft_plan_modified_hermite_to_hermite.
ft_modified_planl * ft_plan_modified_hermite_to_hermitel(const int n, const int nu, const long double * u, const int nv, const long double * v, const int verbose);

#include <mpfr.h>

typedef struct {
    mpfr_t * data;
    int n;
    int b;
} ft_mpfr_triangular_banded;

void ft_mpfr_destroy_plan(mpfr_t * A, int n);
void ft_mpfr_trmv(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t * x, mpfr_rnd_t rnd);
void ft_mpfr_trsv(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t * x, mpfr_rnd_t rnd);
void ft_mpfr_trmm(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t * B, int LDB, int N, mpfr_rnd_t rnd);
void ft_mpfr_trsm(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t * B, int LDB, int N, mpfr_rnd_t rnd);
// C -- Julia interoperability. Julia `BigFloat` does not have the same size as `mpfr_t`.
// So we give all Julia-owned data its own address, and dereference to retrieve the number.
void ft_mpfr_trmv_ptr(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t ** x, mpfr_rnd_t rnd);
void ft_mpfr_trsv_ptr(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t ** x, mpfr_rnd_t rnd);
void ft_mpfr_trmm_ptr(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t ** B, int LDB, int N, mpfr_rnd_t rnd);
void ft_mpfr_trsm_ptr(char TRANS, int n, mpfr_t * A, int LDA, mpfr_t ** B, int LDB, int N, mpfr_rnd_t rnd);

/*!
  \brief A multi-precision version of \ref ft_plan_legendre_to_chebyshev that returns a dense array of connection coefficients.
  \param normleg The normalization of the Legendre polynomials: standard (`0`) or orthonormalized (`1`).
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_chebyshev_to_legendre that returns a dense array of connection coefficients.
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param normleg The normalization of the Legendre polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_ultraspherical_to_ultraspherical that returns a dense array of connection coefficients.
  \param norm1 The normalization of the source ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param norm2 The normalization of the target ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the source ultraspherical polynomials.
  \param mu The parameter \f$\mu\f$ of the target ultraspherical polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, mpfr_srcptr lambda, mpfr_srcptr mu, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_jacobi_to_jacobi that returns a dense array of connection coefficients.
  \param norm1 The normalization of the source Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param norm2 The normalization of the target Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the source Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the source Jacobi polynomials.
  \param gamma The parameter \f$\gamma\f$ of the target Jacobi polynomials, in the place of \f$\alpha\f$.
  \param delta The parameter \f$\delta\f$ of the target Jacobi polynomials, in the place of \f$\beta\f$.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr gamma, mpfr_srcptr delta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_laguerre_to_laguerre that returns a dense array of connection coefficients.
  \param norm1 The normalization of the source Laguerre polynomials: standard (`0`) or orthonormalized (`1`).
  \param norm2 The normalization of the target Laguerre polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the source Laguerre polynomials.
  \param beta The parameter \f$\beta\f$ of the target Laguerre polynomials, in the place of \f$\alpha\f$.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_jacobi_to_ultraspherical that returns a dense array of connection coefficients.
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_ultraspherical_to_jacobi that returns a dense array of connection coefficients.
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, mpfr_srcptr lambda, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_jacobi_to_chebyshev that returns a dense array of connection coefficients.
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_chebyshev_to_jacobi that returns a dense array of connection coefficients.
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param normjac The normalization of the Jacobi polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ of the Jacobi polynomials.
  \param beta The parameter \f$\beta\f$ of the Jacobi polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_ultraspherical_to_chebyshev that returns a dense array of connection coefficients.
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd);
/*!
  \brief A multi-precision version of \ref ft_plan_chebyshev_to_ultraspherical that returns a dense array of connection coefficients.
  \param normcheb The normalization of the Chebyshev polynomials: standard (`0`) or orthonormalized (`1`).
  \param normultra The normalization of the ultraspherical polynomials: standard (`0`) or orthonormalized (`1`).
  \param n The number of coefficients in each expansion, so that the polynomial degree is at most \f$n-1\f$.
  \param lambda The parameter \f$\lambda\f$ of the ultraspherical polynomials.
  \param prec The precision of the connection coefficients, in bits.
  \param rnd The MPFR rounding mode.
*/
mpfr_t * ft_mpfr_plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd);

/*!
  \brief Set the number of OpenMP threads.
  \param n The number of threads.
*/
void ft_set_num_threads(const int n);

/// Data structure to store sines and cosines of Givens rotations.
typedef struct ft_rotation_plan_s ft_rotation_plan;

/// Destroy a \ref ft_rotation_plan.
void ft_destroy_rotation_plan(ft_rotation_plan * RP);

ft_rotation_plan * ft_plan_rotsphere(const int n);

/// Convert a single vector of spherical harmonic coefficients in A with stride S from order m2 down to order m1.
void ft_kernel_sph_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
/// Convert a single vector of spherical harmonic coefficients in A with stride S from order m1 up to order m2.
void ft_kernel_sph_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

ft_rotation_plan * ft_plan_rottriangle(const int n, const double alpha, const double beta, const double gamma);

/// Convert a single vector of triangular harmonic coefficients in A with stride S from order m2 down to order m1.
void ft_kernel_tri_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
/// Convert a single vector of triangular harmonic coefficients in A with stride S from order m1 up to order m2.
void ft_kernel_tri_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

ft_rotation_plan * ft_plan_rotdisk(const int n, const double alpha, const double beta);
ft_rotation_plan * ft_plan_rotannulus(const int n, const double alpha, const double beta, const double gamma, const double rho);

/// Convert a single vector of disk harmonic coefficients in A with stride S from order m2 down to order m1.
void ft_kernel_disk_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
/// Convert a single vector of disk harmonic coefficients in A with stride S from order m1 up to order m2.
void ft_kernel_disk_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

ft_rotation_plan * ft_plan_rotrectdisk(const int n, const double beta);

/// Convert a single vector of rectangularized disk harmonic coefficients in A with stride S from order m2 down to order m1.
void ft_kernel_rectdisk_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
/// Convert a single vector of rectangularized disk harmonic coefficients in A with stride S from order m1 up to order m2.
void ft_kernel_rectdisk_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

void ft_kernel_tet_hi2lo(const ft_rotation_plan * RP, const int L, const int m, double * A);
void ft_kernel_tet_lo2hi(const ft_rotation_plan * RP, const int L, const int m, double * A);

/// Data structure to store sines and cosines of Givens rotations for spin-weighted harmonics.
typedef struct ft_spin_rotation_plan_s ft_spin_rotation_plan;

/// Destroy a \ref ft_spin_rotation_plan.
void ft_destroy_spin_rotation_plan(ft_spin_rotation_plan * SRP);

ft_spin_rotation_plan * ft_plan_rotspinsphere(const int n, const int s);

/// Convert a single vector of spin-weighted spherical harmonic coefficients in A with stride S from order m down to order m%2.
void ft_kernel_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);
/// Convert a single vector of spin-weighted spherical harmonic coefficients in A with stride S from order m%2 up to order m.
void ft_kernel_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, const int m, ft_complex * A, const int S);


void ft_execute_sph_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sph_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_sphv_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sphv_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_tri_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_tri_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_disk_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_disk_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_rectdisk_hi2lo(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_rectdisk_lo2hi(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_tet_hi2lo(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M);
void ft_execute_tet_lo2hi(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M);

void ft_execute_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);
void ft_execute_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);

/// Data structure to store \ref ft_rotation_plan "ft_rotation_plan"s, arrays to represent 1D orthogonal polynomial transforms and their inverses, and Greek parameters.
typedef struct {
    ft_rotation_plan ** RP;
    ft_modified_plan ** MP;
    double * B;
    double ** P;
    double ** Pinv;
    double alpha;
    double beta;
    double gamma;
    double delta;
    double rho;
    int NRP;
    int NMP;
    int NP;
} ft_harmonic_plan;

/// Destroy a \ref ft_harmonic_plan.
void ft_destroy_harmonic_plan(ft_harmonic_plan * P);

/*!
  \brief Plan a spherical harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$n-1\f$.
*/
ft_harmonic_plan * ft_plan_sph2fourier(const int n);

/*!
  \brief Transform a spherical harmonic expansion to a bivariate Fourier series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_sph2fourier.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref sph2fourier "sph2fourier". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$2N-1\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_sph2fourier(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);
/*!
  \brief Transform a bivariate Fourier series to a spherical harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_sph2fourier.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref sph2fourier "sph2fourier". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$2N-1\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_fourier2sph(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);

void ft_execute_sphv2fourier(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);
void ft_execute_fourier2sphv(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);

/*!
  \brief Plan a triangular harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ in the weight \f$x^\alpha y^\beta(1-x-y)^\gamma\f$.
  \param beta The parameter \f$\beta\f$ in the weight \f$x^\alpha y^\beta(1-x-y)^\gamma\f$.
  \param gamma The parameter \f$\gamma\f$ in the weight \f$x^\alpha y^\beta(1-x-y)^\gamma\f$.
*/
ft_harmonic_plan * ft_plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma);

/*!
  \brief Transform a triangular harmonic expansion to a bivariate Chebyshev series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_tri2cheb.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref tri2cheb "tri2cheb". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_tri2cheb(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);
/*!
  \brief Transform a bivariate Chebyshev series to a triangular harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_tri2cheb.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref tri2cheb "tri2cheb". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_cheb2tri(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);

/*!
  \brief Plan a disk harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$2n-2\f$.
  \param alpha The parameter \f$\alpha\f$ in the weight \f$r^{2\alpha+1}(1-r^2)^\beta\f$.
  \param beta The parameter \f$\beta\f$ in the weight \f$r^{2\alpha+1}(1-r^2)^\beta\f$.
*/
ft_harmonic_plan * ft_plan_disk2cxf(const int n, const double alpha, const double beta);

/*!
  \brief Transform a disk harmonic expansion to a Chebyshev--Fourier series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_disk2cxf.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref disk2cxf "disk2cxf". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$4N-3\f$ for an expansion of degree \f$2N-2\f$.
*/
void ft_execute_disk2cxf(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);
/*!
  \brief Transform a Chebyshev--Fourier series to a disk harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_disk2cxf.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref disk2cxf "disk2cxf". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$4N-3\f$ for an expansion of degree \f$2N-2\f$.
*/
void ft_execute_cxf2disk(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);

/*!
  \brief Plan an annulus harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$2n-2\f$.
  \param alpha The parameter \f$\alpha\f$ in the weight \f$r^{2\gamma+1}(r^2-\rho^2)^\alpha(1-r^2)^\beta\f$.
  \param beta The parameter \f$\beta\f$ in the weight \f$r^{2\gamma+1}(r^2-\rho^2)^\alpha(1-r^2)^\beta\f$.
  \param gamma The parameter \f$\gamma\f$ in the weight \f$r^{2\gamma+1}(r^2-\rho^2)^\alpha(1-r^2)^\beta\f$.
  \param rho The inner radius of the annulus, \f$0<\rho<1\f$.
*/
ft_harmonic_plan * ft_plan_ann2cxf(const int n, const double alpha, const double beta, const double gamma, const double rho);

/*!
  \brief Transform an annulus harmonic expansion to a Chebyshev--Fourier series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_ann2cxf.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref ann2cxf "ann2cxf". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$4N-3\f$ for an expansion of degree \f$2N-2\f$.
*/
void ft_execute_ann2cxf(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);
/*!
  \brief Transform a Chebyshev--Fourier series to an annulus harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_ann2cxf.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref ann2cxf "ann2cxf". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$4N-3\f$ for an expansion of degree \f$2N-2\f$.
*/
void ft_execute_cxf2ann(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);

/*!
  \brief Plan a rectangularized disk harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$n-1\f$.
  \param beta The parameter \f$\beta\f$ in the weight \f$(1-x^2-y^2)^\beta\f$.
*/
ft_harmonic_plan * ft_plan_rectdisk2cheb(const int n, const double beta);

/*!
  \brief Transform a rectangularized disk harmonic expansion to a Chebyshev series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_rectdisk2cheb.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref rectdisk2cheb "rectdisk2cheb". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_rectdisk2cheb(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);
/*!
  \brief Transform a Chebyshev series to a rectangularized disk harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_rectdisk2cheb.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref rectdisk2cheb "rectdisk2cheb". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_cheb2rectdisk(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int M);

/*!
  \brief Plan a tetrahedral harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$n-1\f$.
  \param alpha The parameter \f$\alpha\f$ in the weight \f$x^\alpha y^\beta z^\gamma(1-x-y-z)^\delta\f$.
  \param beta The parameter \f$\beta\f$ in the weight \f$x^\alpha y^\beta z^\gamma(1-x-y-z)^\delta\f$.
  \param gamma The parameter \f$\gamma\f$ in the weight \f$x^\alpha y^\beta z^\gamma(1-x-y-z)^\delta\f$.
  \param delta The parameter \f$\delta\f$ in the weight \f$x^\alpha y^\beta z^\gamma(1-x-y-z)^\delta\f$.
*/
ft_harmonic_plan * ft_plan_tet2cheb(const int n, const double alpha, const double beta, const double gamma, const double delta);

/*!
  \brief Transform a tetrahedral harmonic expansion to a trivariate Chebyshev series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_tet2cheb.
  \param A The \f$N\times L\times M\f$ array of coefficients in column-major ordering, as in \ref tet2cheb "tet2cheb". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param L The number of columns of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
  \param M The number of slices of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_tet2cheb(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int L, const int M);
/*!
  \brief Transform a trivariate Chebyshev series to a tetrahedral harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_tet2cheb.
  \param A The \f$N\times L\times M\f$ array of coefficients in column-major ordering, as in \ref tet2cheb "tet2cheb". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param L The number of columns of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
  \param M The number of slices of `A`, which is \f$N\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_cheb2tet(const char TRANS, const ft_harmonic_plan * P, double * A, const int N, const int L, const int M);

/// Data structure to store a \ref ft_spin_rotation_plan, and various arrays to represent 1D orthogonal polynomial transforms.
typedef struct {
    ft_spin_rotation_plan * SRP;
    ft_complex * B;
    ft_complex * P1;
    ft_complex * P2;
    ft_complex * P1inv;
    ft_complex * P2inv;
    int s;
} ft_spin_harmonic_plan;

/// Destroy a \ref ft_spin_harmonic_plan.
void ft_destroy_spin_harmonic_plan(ft_spin_harmonic_plan * P);

/*!
  \brief Plan a spin-weighted spherical harmonic transform.
  \param n The number of rows in the array of coefficients, so that the polynomial degree is at most \f$n-1\f$.
  \param s The integer spin \f$s\f$.
*/
ft_spin_harmonic_plan * ft_plan_spinsph2fourier(const int n, const int s);

/*!
  \brief Transform a spin-weighted spherical harmonic expansion to a bivariate Fourier series.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_spinsph2fourier.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref spinsph2fourier "spinsph2fourier". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$2N-1\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_spinsph2fourier(const char TRANS, const ft_spin_harmonic_plan * P, ft_complex * A, const int N, const int M);
/*!
  \brief Transform a bivariate Fourier series to a spin-weighted spherical harmonic expansion.
  \param TRANS `'N'` to apply the transform or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_spinsph2fourier.
  \param A The \f$N\times M\f$ array of coefficients in column-major ordering, as in \ref spinsph2fourier "spinsph2fourier". It is overwritten by the result.
  \param N The number of rows of `A`, equal to the `n` that created the plan.
  \param M The number of columns of `A`, which is \f$2N-1\f$ for an expansion of degree \f$N-1\f$.
*/
void ft_execute_fourier2spinsph(const char TRANS, const ft_spin_harmonic_plan * P, ft_complex * A, const int N, const int M);


int ft_fftw_init_threads(void);
void ft_fftw_plan_with_nthreads(const int n);

typedef struct {
    fftw_plan plantheta1;
    fftw_plan plantheta2;
    fftw_plan plantheta3;
    fftw_plan plantheta4;
    fftw_plan planphi;
    double * Y;
} ft_sphere_fftw_plan;

/// Destroy a \ref ft_sphere_fftw_plan.
void ft_destroy_sphere_fftw_plan(ft_sphere_fftw_plan * P);

ft_sphere_fftw_plan * ft_plan_sph_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1], const unsigned flags);
/*!
  \brief Plan FFTW synthesis on the sphere.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_sphere_fftw_plan * ft_plan_sph_synthesis(const int N, const int M, const unsigned flags);
/*!
  \brief Plan FFTW analysis on the sphere.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_sphere_fftw_plan * ft_plan_sph_analysis(const int N, const int M, const unsigned flags);
ft_sphere_fftw_plan * ft_plan_sphv_synthesis(const int N, const int M, const unsigned flags);
ft_sphere_fftw_plan * ft_plan_sphv_analysis(const int N, const int M, const unsigned flags);

/*!
  \brief Execute FFTW synthesis on the sphere.
  \param TRANS `'N'` to apply the synthesis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_sph_synthesis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the coefficients on entry and the function values on the grid on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_sph_synthesis(const char TRANS, const ft_sphere_fftw_plan * P, double * X, const int N, const int M);
/*!
  \brief Execute FFTW analysis on the sphere.
  \param TRANS `'N'` to apply the analysis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_sph_analysis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the function values on the grid on entry and the coefficients on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_sph_analysis(const char TRANS, const ft_sphere_fftw_plan * P, double * X, const int N, const int M);

void ft_execute_sphv_synthesis(const char TRANS, const ft_sphere_fftw_plan * P, double * X, const int N, const int M);
void ft_execute_sphv_analysis(const char TRANS, const ft_sphere_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planxy;
} ft_triangle_fftw_plan;

/// Destroy a \ref ft_triangle_fftw_plan.
void ft_destroy_triangle_fftw_plan(ft_triangle_fftw_plan * P);

ft_triangle_fftw_plan * ft_plan_tri_with_kind(const int N, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1, const unsigned flags);
/*!
  \brief Plan FFTW synthesis on the triangle.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_triangle_fftw_plan * ft_plan_tri_synthesis(const int N, const int M, const unsigned flags);
/*!
  \brief Plan FFTW analysis on the triangle.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_triangle_fftw_plan * ft_plan_tri_analysis(const int N, const int M, const unsigned flags);

/*!
  \brief Execute FFTW synthesis on the triangle.
  \param TRANS `'N'` to apply the synthesis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_tri_synthesis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the coefficients on entry and the function values on the grid on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_tri_synthesis(const char TRANS, const ft_triangle_fftw_plan * P, double * X, const int N, const int M);
/*!
  \brief Execute FFTW analysis on the triangle.
  \param TRANS `'N'` to apply the analysis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_tri_analysis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the function values on the grid on entry and the coefficients on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_tri_analysis(const char TRANS, const ft_triangle_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planxyz;
} ft_tetrahedron_fftw_plan;

void ft_destroy_tetrahedron_fftw_plan(ft_tetrahedron_fftw_plan * P);

ft_tetrahedron_fftw_plan * ft_plan_tet_with_kind(const int N, const int L, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1, const fftw_r2r_kind kind2, const unsigned flags);
ft_tetrahedron_fftw_plan * ft_plan_tet_synthesis(const int N, const int L, const int M, const unsigned flags);
ft_tetrahedron_fftw_plan * ft_plan_tet_analysis(const int N, const int L, const int M, const unsigned flags);

void ft_execute_tet_synthesis(const char TRANS, const ft_tetrahedron_fftw_plan * P, double * X, const int N, const int L, const int M);
void ft_execute_tet_analysis(const char TRANS, const ft_tetrahedron_fftw_plan * P, double * X, const int N, const int L, const int M);

typedef struct {
    fftw_plan planr1;
    fftw_plan planr2;
    fftw_plan planr3;
    fftw_plan planr4;
    fftw_plan plantheta;
    double * Y;
} ft_disk_fftw_plan;

/// Destroy a \ref ft_disk_fftw_plan.
void ft_destroy_disk_fftw_plan(ft_disk_fftw_plan * P);

ft_disk_fftw_plan * ft_plan_disk_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1], const unsigned flags);
/*!
  \brief Plan FFTW synthesis on the disk.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_disk_fftw_plan * ft_plan_disk_synthesis(const int N, const int M, const unsigned flags);
/*!
  \brief Plan FFTW analysis on the disk.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_disk_fftw_plan * ft_plan_disk_analysis(const int N, const int M, const unsigned flags);

/*!
  \brief Execute FFTW synthesis on the disk.
  \param TRANS `'N'` to apply the synthesis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_disk_synthesis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the coefficients on entry and the function values on the grid on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_disk_synthesis(const char TRANS, const ft_disk_fftw_plan * P, double * X, const int N, const int M);
/*!
  \brief Execute FFTW analysis on the disk.
  \param TRANS `'N'` to apply the analysis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_disk_analysis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the function values on the grid on entry and the coefficients on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_disk_analysis(const char TRANS, const ft_disk_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planr;
    fftw_plan plantheta;
    double rho;
    double * w;
    double * Y;
} ft_annulus_fftw_plan;

/// Destroy a \ref ft_annulus_fftw_plan.
void ft_destroy_annulus_fftw_plan(ft_annulus_fftw_plan * P);

double ft_get_rho_annulus_fftw_plan(ft_annulus_fftw_plan * P);

ft_annulus_fftw_plan * ft_plan_annulus_with_kind(const int N, const int M, const double rho, const fftw_r2r_kind kind[3][1], const unsigned flags);
/*!
  \brief Plan FFTW synthesis on the annulus.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param rho The inner radius of the annulus, \f$0<\rho<1\f$.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_annulus_fftw_plan * ft_plan_annulus_synthesis(const int N, const int M, const double rho, const unsigned flags);
/*!
  \brief Plan FFTW analysis on the annulus.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param rho The inner radius of the annulus, \f$0<\rho<1\f$.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_annulus_fftw_plan * ft_plan_annulus_analysis(const int N, const int M, const double rho, const unsigned flags);

/*!
  \brief Execute FFTW synthesis on the annulus.
  \param TRANS `'N'` to apply the synthesis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_annulus_synthesis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the coefficients on entry and the function values on the grid on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_annulus_synthesis(const char TRANS, const ft_annulus_fftw_plan * P, double * X, const int N, const int M);
/*!
  \brief Execute FFTW analysis on the annulus.
  \param TRANS `'N'` to apply the analysis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_annulus_analysis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the function values on the grid on entry and the coefficients on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_annulus_analysis(const char TRANS, const ft_annulus_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planx1;
    fftw_plan planx2;
    fftw_plan plany;
} ft_rectdisk_fftw_plan;

/// Destroy a \ref ft_rectdisk_fftw_plan.
void ft_destroy_rectdisk_fftw_plan(ft_rectdisk_fftw_plan * P);

ft_rectdisk_fftw_plan * ft_plan_rectdisk_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1], const unsigned flags);
/*!
  \brief Plan FFTW synthesis on the rectangularized disk.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_rectdisk_fftw_plan * ft_plan_rectdisk_synthesis(const int N, const int M, const unsigned flags);
/*!
  \brief Plan FFTW analysis on the rectangularized disk.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_rectdisk_fftw_plan * ft_plan_rectdisk_analysis(const int N, const int M, const unsigned flags);

/*!
  \brief Execute FFTW synthesis on the rectangularized disk.
  \param TRANS `'N'` to apply the synthesis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_rectdisk_synthesis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the coefficients on entry and the function values on the grid on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_rectdisk_synthesis(const char TRANS, const ft_rectdisk_fftw_plan * P, double * X, const int N, const int M);
/*!
  \brief Execute FFTW analysis on the rectangularized disk.
  \param TRANS `'N'` to apply the analysis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_rectdisk_analysis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the function values on the grid on entry and the coefficients on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_rectdisk_analysis(const char TRANS, const ft_rectdisk_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan plantheta1;
    fftw_plan plantheta2;
    fftw_plan plantheta3;
    fftw_plan plantheta4;
    fftw_plan planphi;
    double * Y;
    int S;
} ft_spinsphere_fftw_plan;

/// Destroy a \ref ft_spinsphere_fftw_plan.
void ft_destroy_spinsphere_fftw_plan(ft_spinsphere_fftw_plan * P);

int ft_get_spin_spinsphere_fftw_plan(const ft_spinsphere_fftw_plan * P);

ft_spinsphere_fftw_plan * ft_plan_spinsph_with_kind(const int N, const int M, const int S, const fftw_r2r_kind kind[2][1], const int sign, const unsigned flags);
/*!
  \brief Plan FFTW synthesis on the sphere with spin.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param S The integer spin.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_spinsphere_fftw_plan * ft_plan_spinsph_synthesis(const int N, const int M, const int S, const unsigned flags);
/*!
  \brief Plan FFTW analysis on the sphere with spin.
  \param N The number of rows of the \f$N\times M\f$ grid, see \ref synthesisanalysis.
  \param M The number of columns of the \f$N\times M\f$ grid.
  \param S The integer spin.
  \param flags The FFTW planner flags, such as `FFTW_ESTIMATE` or `FFTW_MEASURE`.
*/
ft_spinsphere_fftw_plan * ft_plan_spinsph_analysis(const int N, const int M, const int S, const unsigned flags);

/*!
  \brief Execute FFTW synthesis on the sphere with spin.
  \param TRANS `'N'` to apply the synthesis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_spinsph_synthesis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the coefficients on entry and the function values on the grid on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_spinsph_synthesis(const char TRANS, const ft_spinsphere_fftw_plan * P, ft_complex * X, const int N, const int M);
/*!
  \brief Execute FFTW analysis on the sphere with spin.
  \param TRANS `'N'` to apply the analysis or `'T'` to apply its transpose.
  \param P The plan that is returned by \ref ft_plan_spinsph_analysis.
  \param X The \f$N\times M\f$ array in column-major ordering that stores the function values on the grid on entry and the coefficients on exit.
  \param N The number of rows of `X`.
  \param M The number of columns of `X`.
*/
void ft_execute_spinsph_analysis(const char TRANS, const ft_spinsphere_fftw_plan * P, ft_complex * X, const int N, const int M);

typedef struct {
    ft_banded ** B;
    ft_triangular_banded ** T;
    int n;
} ft_gradient_plan;

void ft_destroy_gradient_plan(ft_gradient_plan * P);

ft_gradient_plan * ft_plan_sph_gradient(const int n);

void ft_execute_sph_gradient(ft_gradient_plan * P, double * U, double * Ut, double * Up, const int N, const int M);
void ft_execute_sph_curl(ft_gradient_plan * P, double * U, double * Ut, double * Up, const int N, const int M);

typedef struct {
    ft_triangular_banded ** T;
    ft_banded_qr ** F;
    double * X;
    int n;
} ft_helmholtzhodge_plan;

void ft_destroy_helmholtzhodge_plan(ft_helmholtzhodge_plan * P);

ft_helmholtzhodge_plan * ft_plan_sph_helmholtzhodge(const int n);

void ft_execute_sph_helmholtzhodge(ft_helmholtzhodge_plan * P, double * U1, double * U2, double * V1, double * V2, const int N, const int M);

/*!
  \brief A static struct to store an orthogonal matrix \f$Q \in \mathbb{R}^{3\times3}\f$, such that \f$Q^\top Q = I\f$.
  \f$Q\f$ has column-major storage.
*/
typedef struct {
    double Q[9];
} ft_orthogonal_transformation;

/*!
  \brief Every orthogonal matrix \f$Q \in \mathbb{R}^{3\times3}\f$ can be decomposed as a product of \f$ZYZ\f$ Euler angles and, if necessary, a reflection \f$R\f$ about the \f$xy\f$-plane.
  \f[
  Q = ZYZR,
  \f]
  where the \f$z\f$-axis rotations are:
  \f[
  Z = \begin{pmatrix} c & -s & 0\\ s & c & 0\\ 0 & 0 & 1\end{pmatrix},
  \f]
  the \f$y\f$-axis rotation is:
  \f[
  Y = \begin{pmatrix} c & 0 & -s\\ 0 & 1 & 0\\ s & 0 & c\end{pmatrix},
  \f]
  and the potential reflection is:
  \f[
  R = \begin{pmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0 & 0 & \pm 1\end{pmatrix}.
  \f]
  The reflection is stored as an integer `sign` corresponding to the bottom right entry.
*/
typedef struct {
    double s[3];
    double c[3];
    int sign;
} ft_ZYZR;

/*!
  \brief A static struct to store a reflection about the plane \f$w\cdot x = 0\f$ in \f$\mathbb{R}^3\f$.
*/
typedef struct {
    double w[3];
} ft_reflection;

ft_ZYZR ft_create_ZYZR(ft_orthogonal_transformation Q);

void ft_apply_ZYZR(ft_ZYZR Q, ft_orthogonal_transformation * U);

void ft_apply_reflection(ft_reflection Q, ft_orthogonal_transformation * U);

void ft_execute_sph_polar_rotation(double * A, const int N, const int M, double s, double c);

void ft_execute_sph_polar_reflection(double * A, const int N, const int M);

typedef struct {
    ft_symmetric_tridiagonal_symmetric_eigen * F11;
    ft_symmetric_tridiagonal_symmetric_eigen * F21;
    ft_symmetric_tridiagonal_symmetric_eigen * F12;
    ft_symmetric_tridiagonal_symmetric_eigen * F22;
    int l;
} ft_partial_sph_isometry_plan;

typedef struct {
    ft_partial_sph_isometry_plan ** F;
    int n;
} ft_sph_isometry_plan;

void ft_destroy_partial_sph_isometry_plan(ft_partial_sph_isometry_plan * F);

void ft_destroy_sph_isometry_plan(ft_sph_isometry_plan * F);

ft_partial_sph_isometry_plan * ft_plan_partial_sph_isometry(const int l);

ft_sph_isometry_plan * ft_plan_sph_isometry(const int n);

void ft_execute_sph_yz_axis_exchange(ft_sph_isometry_plan * J, double * A, const int N, const int M);

void ft_execute_sph_isometry(ft_sph_isometry_plan * J, ft_ZYZR Q, double * A, const int N, const int M);

void ft_execute_sph_rotation(ft_sph_isometry_plan * J, const double alpha, const double beta, const double gamma, double * A, const int N, const int M);

void ft_execute_sph_reflection(ft_sph_isometry_plan * J, ft_reflection W, double * A, const int N, const int M);

void ft_execute_sph_orthogonal_transformation(ft_sph_isometry_plan * J, ft_orthogonal_transformation Q, double * A, const int N, const int M);

#endif // FASTTRANSFORMS_H
