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

#include "tdc.h"

/*!
  \brief Pre-compute a factorization of the connection coefficients between Legendre and Chebyshev polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Legendre}} P_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x).
  \f]
  `normleg` and `normcheb` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_legendre_to_chebyshevf, \ref ft_plan_legendre_to_chebyshevl, and \ref ft_mpfr_plan_legendre_to_chebyshev.
*/
ft_tb_eigen_FMM * ft_plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Chebyshev and Legendre polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Legendre}} P_\ell(x).
  \f]
  `normcheb` and `normleg` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_chebyshev_to_legendref, \ref ft_plan_chebyshev_to_legendrel, and \ref ft_mpfr_plan_chebyshev_to_legendre.
*/
ft_tb_eigen_FMM * ft_plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n);
/*!
  \brief Pre-compute a factorization of the connection coefficients between ultraspherical polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{(1)} C_\ell^{(\lambda)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{(2)} C_\ell^{(\mu)}(x).
  \f]
  `norm1` and `norm2` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_ultraspherical_to_ultrasphericalf, \ref ft_plan_ultraspherical_to_ultrasphericall, and \ref ft_mpfr_plan_ultraspherical_to_ultraspherical.
*/
ft_tb_eigen_FMM * ft_plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, const double lambda, const double mu);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Jacobi polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{(1)} P_\ell^{(\alpha,\beta)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{(2)} P_\ell^{(\gamma,\delta)}(x).
  \f]
  `norm1` and `norm2` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_jacobi_to_jacobif, \ref ft_plan_jacobi_to_jacobil, and \ref ft_mpfr_plan_jacobi_to_jacobi.
*/
ft_tb_eigen_FMM * ft_plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, const double alpha, const double beta, const double gamma, const double delta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Laguerre polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{(1)} L_\ell^{(\alpha)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{(2)} L_\ell^{(\beta)}(x).
  \f]
  `norm1` and `norm2` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_laguerre_to_laguerref, \ref ft_plan_laguerre_to_laguerrel, and \ref ft_mpfr_plan_laguerre_to_laguerre.
*/
ft_tb_eigen_FMM * ft_plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Jacobi and ultraspherical polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x).
  \f]
  `normjac` and `normultra` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_jacobi_to_ultrasphericalf, \ref ft_plan_jacobi_to_ultrasphericall, and \ref ft_mpfr_plan_jacobi_to_ultraspherical.
*/
ft_tb_eigen_FMM * ft_plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, const double alpha, const double beta, const double lambda);
/*!
  \brief Pre-compute a factorization of the connection coefficients between ultraspherical and Jacobi polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x).
  \f]
  `normultra` and `normjac` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_ultraspherical_to_jacobif, \ref ft_plan_ultraspherical_to_jacobil, and \ref ft_mpfr_plan_ultraspherical_to_jacobi.
*/
ft_tb_eigen_FMM * ft_plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, const double lambda, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Jacobi and Chebyshev polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x).
  \f]
  `normjac` and `normcheb` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_jacobi_to_chebyshevf, \ref ft_plan_jacobi_to_chebyshevl, and \ref ft_mpfr_plan_jacobi_to_chebyshev.
*/
ft_tb_eigen_FMM * ft_plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Chebyshev and Jacobi polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Jacobi}} P_\ell^{(\alpha,\beta)}(x).
  \f]
  `normcheb` and `normjac` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_chebyshev_to_jacobif, \ref ft_plan_chebyshev_to_jacobil, and \ref ft_mpfr_plan_chebyshev_to_jacobi.
*/
ft_tb_eigen_FMM * ft_plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, const double alpha, const double beta);
/*!
  \brief Pre-compute a factorization of the connection coefficients between ultraspherical and Chebyshev polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x).
  \f]
  `normultra` and `normcheb` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_ultraspherical_to_chebyshevf, \ref ft_plan_ultraspherical_to_chebyshevl, and \ref ft_mpfr_plan_ultraspherical_to_chebyshev.
*/
ft_tb_eigen_FMM * ft_plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, const double lambda);
/*!
  \brief Pre-compute a factorization of the connection coefficients between Chebyshev and ultraspherical polynomials in double precision so that ft_bfmv converts between expansions:
  \f[
  \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{Chebyshev}} T_\ell(x) = \sum_{\ell=0}^{n-1} c_\ell^{\mathrm{ultraspherical}} C_\ell^{(\lambda)}(x).
  \f]
  `normcheb` and `normultra` govern the normalizations, either standard ( == 0) or orthonormalized ( == 1).\n
  See also \ref ft_plan_chebyshev_to_ultrasphericalf, \ref ft_plan_chebyshev_to_ultrasphericall, and \ref ft_mpfr_plan_chebyshev_to_ultraspherical.
*/
ft_tb_eigen_FMM * ft_plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, const double lambda);

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

/// A multi-precision version of \ref ft_plan_legendre_to_chebyshev that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_chebyshev_to_legendre that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_ultraspherical_to_ultraspherical that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_ultraspherical_to_ultraspherical(const int norm1, const int norm2, const int n, mpfr_srcptr lambda, mpfr_srcptr mu, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_jacobi_to_jacobi that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_jacobi_to_jacobi(const int norm1, const int norm2, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr gamma, mpfr_srcptr delta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_laguerre_to_laguerre that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_laguerre_to_laguerre(const int norm1, const int norm2, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_jacobi_to_ultraspherical that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_jacobi_to_ultraspherical(const int normjac, const int normultra, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_ultraspherical_to_jacobi that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_ultraspherical_to_jacobi(const int normultra, const int normjac, const int n, mpfr_srcptr lambda, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_jacobi_to_chebyshev that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_jacobi_to_chebyshev(const int normjac, const int normcheb, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_chebyshev_to_jacobi that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_chebyshev_to_jacobi(const int normcheb, const int normjac, const int n, mpfr_srcptr alpha, mpfr_srcptr beta, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_ultraspherical_to_chebyshev that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_ultraspherical_to_chebyshev(const int normultra, const int normcheb, const int n, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd);
/// A multi-precision version of \ref ft_plan_chebyshev_to_ultraspherical that returns a dense array of connection coefficients.
mpfr_t * ft_mpfr_plan_chebyshev_to_ultraspherical(const int normcheb, const int normultra, const int n, mpfr_srcptr lambda, mpfr_prec_t prec, mpfr_rnd_t rnd);

/// Set the number of OpenMP threads.
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

ft_rotation_plan * ft_plan_rotdisk(const int n);

/// Convert a single vector of disk harmonic coefficients in A with stride S from order m2 down to order m1.
void ft_kernel_disk_hi2lo(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);
/// Convert a single vector of disk harmonic coefficients in A with stride S from order m1 up to order m2.
void ft_kernel_disk_lo2hi(const ft_rotation_plan * RP, const int m1, const int m2, double * A, const int S);

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

void ft_execute_tet_hi2lo(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M);
void ft_execute_tet_lo2hi(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M);

void ft_execute_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);
void ft_execute_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, ft_complex * A, ft_complex * B, const int M);

/// Data structure to store a \ref ft_rotation_plan, and various arrays to represent 1D orthogonal polynomial transforms.
typedef struct {
    ft_rotation_plan * RP;
    double * B;
    double * P1;
    double * P2;
    double * P1inv;
    double * P2inv;
    double alpha;
    double beta;
    double gamma;
} ft_harmonic_plan;

/// Destroy a \ref ft_harmonic_plan.
void ft_destroy_harmonic_plan(ft_harmonic_plan * P);

/// Plan a spherical harmonic transform.
ft_harmonic_plan * ft_plan_sph2fourier(const int n);

/// Transform a spherical harmonic expansion to a bivariate Fourier series.
void ft_execute_sph2fourier(const ft_harmonic_plan * P, double * A, const int N, const int M);
/// Transform a bivariate Fourier series to a spherical harmonic expansion.
void ft_execute_fourier2sph(const ft_harmonic_plan * P, double * A, const int N, const int M);

void ft_execute_sphv2fourier(const ft_harmonic_plan * P, double * A, const int N, const int M);
void ft_execute_fourier2sphv(const ft_harmonic_plan * P, double * A, const int N, const int M);

/// Plan a triangular harmonic transform.
ft_harmonic_plan * ft_plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma);

/// Transform a triangular harmonic expansion to a bivariate Chebyshev series.
void ft_execute_tri2cheb(const ft_harmonic_plan * P, double * A, const int N, const int M);
/// Transform a bivariate Chebyshev series to a triangular harmonic expansion.
void ft_execute_cheb2tri(const ft_harmonic_plan * P, double * A, const int N, const int M);

/// Plan a disk harmonic transform.
ft_harmonic_plan * ft_plan_disk2cxf(const int n);

/// Transform a disk harmonic expansion to a Chebyshev--Fourier series.
void ft_execute_disk2cxf(const ft_harmonic_plan * P, double * A, const int N, const int M);
/// Transform a Chebyshev--Fourier series to a disk harmonic expansion.
void ft_execute_cxf2disk(const ft_harmonic_plan * P, double * A, const int N, const int M);

typedef struct {
    ft_rotation_plan * RP1;
    ft_rotation_plan * RP2;
    double * B;
    double * P1;
    double * P2;
    double * P3;
    double * P1inv;
    double * P2inv;
    double * P3inv;
    double alpha;
    double beta;
    double gamma;
    double delta;
} ft_tetrahedral_harmonic_plan;

void ft_destroy_tetrahedral_harmonic_plan(ft_tetrahedral_harmonic_plan * P);

/// Plan a tetrahedral harmonic transform.
ft_tetrahedral_harmonic_plan * ft_plan_tet2cheb(const int n, const double alpha, const double beta, const double gamma, const double delta);

/// Transform a tetrahedral harmonic expansion to a trivariate Chebyshev series.
void ft_execute_tet2cheb(const ft_tetrahedral_harmonic_plan * P, double * A, const int N, const int L, const int M);
/// Transform a trivariate Chebyshev series to a tetrahedral harmonic expansion.
void ft_execute_cheb2tet(const ft_tetrahedral_harmonic_plan * P, double * A, const int N, const int L, const int M);

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

/// Plan a spin-weighted spherical harmonic transform.
ft_spin_harmonic_plan * ft_plan_spinsph2fourier(const int n, const int s);

/// Transform a spin-weighted spherical harmonic expansion to a bivariate Fourier series.
void ft_execute_spinsph2fourier(const ft_spin_harmonic_plan * P, ft_complex * A, const int N, const int M);
/// Transform a bivariate Fourier series to a spin-weighted spherical harmonic expansion.
void ft_execute_fourier2spinsph(const ft_spin_harmonic_plan * P, ft_complex * A, const int N, const int M);


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

ft_sphere_fftw_plan * ft_plan_sph_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1]);
/// Plan FFTW synthesis on the sphere.
ft_sphere_fftw_plan * ft_plan_sph_synthesis(const int N, const int M);
/// Plan FFTW analysis on the sphere.
ft_sphere_fftw_plan * ft_plan_sph_analysis(const int N, const int M);
ft_sphere_fftw_plan * ft_plan_sphv_synthesis(const int N, const int M);
ft_sphere_fftw_plan * ft_plan_sphv_analysis(const int N, const int M);

/// Execute FFTW synthesis on the sphere.
void ft_execute_sph_synthesis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);
/// Execute FFTW analysis on the sphere.
void ft_execute_sph_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);

void ft_execute_sphv_synthesis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);
void ft_execute_sphv_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planxy;
} ft_triangle_fftw_plan;

/// Destroy a \ref ft_triangle_fftw_plan.
void ft_destroy_triangle_fftw_plan(ft_triangle_fftw_plan * P);

ft_triangle_fftw_plan * ft_plan_tri_with_kind(const int N, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1);
/// Plan FFTW synthesis on the triangle.
ft_triangle_fftw_plan * ft_plan_tri_synthesis(const int N, const int M);
/// Plan FFTW analysis on the triangle.
ft_triangle_fftw_plan * ft_plan_tri_analysis(const int N, const int M);

/// Execute FFTW synthesis on the triangle.
void ft_execute_tri_synthesis(const ft_triangle_fftw_plan * P, double * X, const int N, const int M);
/// Execute FFTW analysis on the triangle.
void ft_execute_tri_analysis(const ft_triangle_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planxyz;
} ft_tetrahedron_fftw_plan;

void ft_destroy_tetrahedron_fftw_plan(ft_tetrahedron_fftw_plan * P);

ft_tetrahedron_fftw_plan * ft_plan_tet_with_kind(const int N, const int L, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1, const fftw_r2r_kind kind2);
ft_tetrahedron_fftw_plan * ft_plan_tet_synthesis(const int N, const int L, const int M);
ft_tetrahedron_fftw_plan * ft_plan_tet_analysis(const int N, const int L, const int M);

void ft_execute_tet_synthesis(const ft_tetrahedron_fftw_plan * P, double * X, const int N, const int L, const int M);
void ft_execute_tet_analysis(const ft_tetrahedron_fftw_plan * P, double * X, const int N, const int L, const int M);

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

ft_disk_fftw_plan * ft_plan_disk_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1]);
/// Plan FFTW synthesis on the disk.
ft_disk_fftw_plan * ft_plan_disk_synthesis(const int N, const int M);
/// Plan FFTW analysis on the disk.
ft_disk_fftw_plan * ft_plan_disk_analysis(const int N, const int M);

/// Execute FFTW synthesis on the disk.
void ft_execute_disk_synthesis(const ft_disk_fftw_plan * P, double * X, const int N, const int M);
/// Execute FFTW analysis on the disk.
void ft_execute_disk_analysis(const ft_disk_fftw_plan * P, double * X, const int N, const int M);

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

ft_spinsphere_fftw_plan * ft_plan_spinsph_with_kind(const int N, const int M, const int S, const fftw_r2r_kind kind[2][1], const int sign);
/// Plan FFTW synthesis on the sphere with spin.
ft_spinsphere_fftw_plan * ft_plan_spinsph_synthesis(const int N, const int M, const int S);
/// Plan FFTW analysis on the sphere with spin.
ft_spinsphere_fftw_plan * ft_plan_spinsph_analysis(const int N, const int M, const int S);

/// Execute FFTW synthesis on the sphere with spin.
void ft_execute_spinsph_synthesis(const ft_spinsphere_fftw_plan * P, ft_complex * X, const int N, const int M);
/// Execute FFTW analysis on the sphere with spin.
void ft_execute_spinsph_analysis(const ft_spinsphere_fftw_plan * P, ft_complex * X, const int N, const int M);

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
