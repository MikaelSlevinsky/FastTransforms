/*!
 \file fasttransforms.h
 \brief fasttransforms.h is the main header file for FastTransforms.
*/
#ifndef FASTTRANSFORMS_H
#define FASTTRANSFORMS_H

#include <cblas.h>
#include <fftw3.h>

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

#include "tdc.h"

/// Set the number of OpenMP threads.
void ft_set_num_threads(const int n);

/// Data structure to store sines and cosines of Givens rotations.
typedef struct {
    double * s;
    double * c;
    int n;
} ft_rotation_plan;

/// Destroy a \ref ft_rotation_plan.
void ft_destroy_rotation_plan(ft_rotation_plan * RP);

ft_rotation_plan * ft_plan_rotsphere(const int n);

/// Convert a single vector of spherical harmonics of order m to 0/1.
void ft_kernel_sph_hi2lo(const ft_rotation_plan * RP, const int m, double * A);
/// Convert a single vector of spherical harmonics of order 0/1 to m.
void ft_kernel_sph_lo2hi(const ft_rotation_plan * RP, const int m, double * A);

/// Convert a pair of vectors of spherical harmonics of order m to 0/1.
void ft_kernel_sph_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A);
/// Convert a pair of vectors of spherical harmonics of order 0/1 to m.
void ft_kernel_sph_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A);

/// Convert four vectors of spherical harmonics of order m, m, m+2, m+2 to 0/1.
void ft_kernel_sph_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A);
/// Convert four vectors of spherical harmonics of order 0/1 to m, m, m+2, m+2.
void ft_kernel_sph_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A);

/// Convert eight vectors of spherical harmonics of order m, m, m+2, m+2, m+4, m+4, m+6, m+6 to 0/1.
void ft_kernel_sph_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A);
/// Convert eight vectors of spherical harmonics of order 0/1 to m, m, m+2, m+2, m+4, m+4, m+6, m+6.
void ft_kernel_sph_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A);

ft_rotation_plan * ft_plan_rottriangle(const int n, const double alpha, const double beta, const double gamma);

/// Convert a single vector of triangular harmonics of order m to 0.
void ft_kernel_tri_hi2lo(const ft_rotation_plan * RP, const int m, double * A);
/// Convert a single vector of triangular harmonics of order 0 to m.
void ft_kernel_tri_lo2hi(const ft_rotation_plan * RP, const int m, double * A);

/// Convert two vectors of triangular harmonics of order m and m+1 to 0.
void ft_kernel_tri_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A);
/// Convert two vectors of triangular harmonics of order 0 to m and m+1.
void ft_kernel_tri_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A);

/// Convert four vectors of triangular harmonics of order m, m+1, m+2, m+3 to 0.
void ft_kernel_tri_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A);
/// Convert four vectors of triangular harmonics of order 0 to m, m+1, m+2, m+3.
void ft_kernel_tri_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A);

/// Convert eight vectors of triangular harmonics of order m, m+1, m+2, m+3, m+4, m+5, m+6, m+7 to 0.
void ft_kernel_tri_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A);
/// Convert eight vectors of triangular harmonics of order 0 to m, m+1, m+2, m+3, m+4, m+5, m+6, m+7.
void ft_kernel_tri_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A);

ft_rotation_plan * ft_plan_rotdisk(const int n);

/// Convert a single vector of disk harmonics of order m to 0/1.
void ft_kernel_disk_hi2lo(const ft_rotation_plan * RP, const int m, double * A);
/// Convert a single vector of disk harmonics of order 0/1 to m.
void ft_kernel_disk_lo2hi(const ft_rotation_plan * RP, const int m, double * A);

/// Convert a pair of vectors of disk harmonics of order m to 0/1.
void ft_kernel_disk_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A);
/// Convert a pair of vectors of disk harmonics of order 0/1 to m.
void ft_kernel_disk_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A);

/// Convert four vectors of disk harmonics of order m, m, m+2, m+2 to 0/1.
void ft_kernel_disk_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A);
/// Convert four vectors of disk harmonics of order 0/1 to m, m, m+2, m+2.
void ft_kernel_disk_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A);

/// Convert eight vectors of disk harmonics of order m, m, m+2, m+2, m+4, m+4, m+6, m+6 to 0/1.
void ft_kernel_disk_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A);
/// Convert eight vectors of disk harmonics of order 0/1 to m, m, m+2, m+2, m+4, m+4, m+6, m+6.
void ft_kernel_disk_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_tet_hi2lo(const ft_rotation_plan * RP, const int L, const int m, double * A);
void ft_kernel_tet_lo2hi(const ft_rotation_plan * RP, const int L, const int m, double * A);

void ft_kernel_tet_hi2lo_SSE(const ft_rotation_plan * RP, const int L, const int m, double * A);
void ft_kernel_tet_lo2hi_SSE(const ft_rotation_plan * RP, const int L, const int m, double * A);

void ft_kernel_tet_hi2lo_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A);
void ft_kernel_tet_lo2hi_AVX(const ft_rotation_plan * RP, const int L, const int m, double * A);

void ft_kernel_tet_hi2lo_AVX512(const ft_rotation_plan * RP, const int L, const int m, double * A);
void ft_kernel_tet_lo2hi_AVX512(const ft_rotation_plan * RP, const int L, const int m, double * A);

typedef struct {
    double * s1;
    double * c1;
    double * s2;
    double * c2;
    double * s3;
    double * c3;
    int n;
    int s;
} ft_spin_rotation_plan;

void ft_destroy_spin_rotation_plan(ft_spin_rotation_plan * SRP);

ft_spin_rotation_plan * ft_plan_rotspinsphere(const int n, const int s);

void ft_kernel_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, const int m, double * A);
void ft_kernel_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, const int m, double * A);

void ft_kernel_spinsph_hi2lo_SSE(const ft_spin_rotation_plan * SRP, const int m, double * A);
void ft_kernel_spinsph_lo2hi_SSE(const ft_spin_rotation_plan * SRP, const int m, double * A);

void ft_kernel_spinsph_hi2lo_AVX(const ft_spin_rotation_plan * SRP, const int m, double * A);
void ft_kernel_spinsph_lo2hi_AVX(const ft_spin_rotation_plan * SRP, const int m, double * A);

void ft_kernel_spinsph_hi2lo_AVX512(const ft_spin_rotation_plan * SRP, const int m, double * A);
void ft_kernel_spinsph_lo2hi_AVX512(const ft_spin_rotation_plan * SRP, const int m, double * A);


void ft_execute_sph_hi2lo(const ft_rotation_plan * RP, double * A, const int M);
void ft_execute_sph_lo2hi(const ft_rotation_plan * RP, double * A, const int M);

void ft_execute_sph_hi2lo_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sph_lo2hi_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_sph_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sph_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_sph_hi2lo_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sph_lo2hi_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_sphv_hi2lo(const ft_rotation_plan * RP, double * A, const int M);
void ft_execute_sphv_lo2hi(const ft_rotation_plan * RP, double * A, const int M);

void ft_execute_sphv_hi2lo_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sphv_lo2hi_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_sphv_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sphv_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_sphv_hi2lo_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_sphv_lo2hi_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_tri_hi2lo(const ft_rotation_plan * RP, double * A, const int M);
void ft_execute_tri_lo2hi(const ft_rotation_plan * RP, double * A, const int M);

void ft_execute_tri_hi2lo_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_tri_lo2hi_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_tri_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_tri_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_tri_hi2lo_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_tri_lo2hi_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_disk_hi2lo(const ft_rotation_plan * RP, double * A, const int M);
void ft_execute_disk_lo2hi(const ft_rotation_plan * RP, double * A, const int M);

void ft_execute_disk_hi2lo_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_disk_lo2hi_SSE(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_disk_hi2lo_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_disk_lo2hi_AVX(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_disk_hi2lo_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);
void ft_execute_disk_lo2hi_AVX512(const ft_rotation_plan * RP, double * A, double * B, const int M);

void ft_execute_tet_hi2lo(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M);
void ft_execute_tet_lo2hi(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, const int L, const int M);

void ft_execute_tet_hi2lo_SSE(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void ft_execute_tet_lo2hi_SSE(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);

void ft_execute_tet_hi2lo_AVX(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void ft_execute_tet_lo2hi_AVX(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);

void ft_execute_tet_hi2lo_AVX512(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);
void ft_execute_tet_lo2hi_AVX512(const ft_rotation_plan * RP1, const ft_rotation_plan * RP2, double * A, double * B, const int L, const int M);

void ft_execute_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, double * A, const int M);
void ft_execute_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, double * A, const int M);

void ft_execute_spinsph_hi2lo_SSE(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);
void ft_execute_spinsph_lo2hi_SSE(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);

void ft_execute_spinsph_hi2lo_AVX(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);
void ft_execute_spinsph_lo2hi_AVX(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);

void ft_execute_spinsph_hi2lo_AVX512(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);
void ft_execute_spinsph_lo2hi_AVX512(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);

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


#endif //FASTTRANSFORMS_H
