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

#include "adc.h"

void ft_set_num_threads(const int n);

typedef struct {
    double * s;
    double * c;
    int n;
} ft_rotation_plan;

void ft_destroy_rotation_plan(ft_rotation_plan * RP);

ft_rotation_plan * ft_plan_rotsphere(const int n);

void ft_kernel_sph_hi2lo(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_sph_lo2hi(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_sph_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_sph_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_sph_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_sph_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_sph_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_sph_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A);

ft_rotation_plan * ft_plan_rottriangle(const int n, const double alpha, const double beta, const double gamma);

void ft_kernel_tri_hi2lo(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_tri_lo2hi(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_tri_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_tri_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_tri_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_tri_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_tri_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_tri_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A);

ft_rotation_plan * ft_plan_rotdisk(const int n);

void ft_kernel_disk_hi2lo(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_disk_lo2hi(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_disk_hi2lo_SSE(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_disk_lo2hi_SSE(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_disk_hi2lo_AVX(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_disk_lo2hi_AVX(const ft_rotation_plan * RP, const int m, double * A);

void ft_kernel_disk_hi2lo_AVX512(const ft_rotation_plan * RP, const int m, double * A);
void ft_kernel_disk_lo2hi_AVX512(const ft_rotation_plan * RP, const int m, double * A);


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

void ft_execute_spinsph_hi2lo(const ft_spin_rotation_plan * SRP, double * A, const int M);
void ft_execute_spinsph_lo2hi(const ft_spin_rotation_plan * SRP, double * A, const int M);

void ft_execute_spinsph_hi2lo_SSE(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);
void ft_execute_spinsph_lo2hi_SSE(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);

void ft_execute_spinsph_hi2lo_AVX(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);
void ft_execute_spinsph_lo2hi_AVX(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);

void ft_execute_spinsph_hi2lo_AVX512(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);
void ft_execute_spinsph_lo2hi_AVX512(const ft_spin_rotation_plan * SRP, double * A, double * B, const int M);

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

void ft_destroy_harmonic_plan(ft_harmonic_plan * P);

ft_harmonic_plan * ft_plan_sph2fourier(const int n);

void ft_execute_sph2fourier(const ft_harmonic_plan * P, double * A, const int N, const int M);
void ft_execute_fourier2sph(const ft_harmonic_plan * P, double * A, const int N, const int M);

void ft_execute_sphv2fourier(const ft_harmonic_plan * P, double * A, const int N, const int M);
void ft_execute_fourier2sphv(const ft_harmonic_plan * P, double * A, const int N, const int M);

ft_harmonic_plan * ft_plan_tri2cheb(const int n, const double alpha, const double beta, const double gamma);

void ft_execute_tri2cheb(const ft_harmonic_plan * P, double * A, const int N, const int M);
void ft_execute_cheb2tri(const ft_harmonic_plan * P, double * A, const int N, const int M);

ft_harmonic_plan * ft_plan_disk2cxf(const int n);

void ft_execute_disk2cxf(const ft_harmonic_plan * P, double * A, const int N, const int M);
void ft_execute_cxf2disk(const ft_harmonic_plan * P, double * A, const int N, const int M);


typedef struct {
    fftw_plan plantheta1;
    fftw_plan plantheta2;
    fftw_plan plantheta3;
    fftw_plan plantheta4;
    fftw_plan planphi;
    double * Y;
} ft_sphere_fftw_plan;

void ft_destroy_sphere_fftw_plan(ft_sphere_fftw_plan * P);

ft_sphere_fftw_plan * ft_plan_sph_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1]);
ft_sphere_fftw_plan * ft_plan_sph_synthesis(const int N, const int M);
ft_sphere_fftw_plan * ft_plan_sph_analysis(const int N, const int M);
ft_sphere_fftw_plan * ft_plan_sphv_synthesis(const int N, const int M);
ft_sphere_fftw_plan * ft_plan_sphv_analysis(const int N, const int M);

void ft_execute_sph_synthesis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);
void ft_execute_sph_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);

void ft_execute_sphv_synthesis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);
void ft_execute_sphv_analysis(const ft_sphere_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planxy;
} ft_triangle_fftw_plan;

void ft_destroy_triangle_fftw_plan(ft_triangle_fftw_plan * P);

ft_triangle_fftw_plan * ft_plan_tri_with_kind(const int N, const int M, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1);
ft_triangle_fftw_plan * ft_plan_tri_synthesis(const int N, const int M);
ft_triangle_fftw_plan * ft_plan_tri_analysis(const int N, const int M);

void ft_execute_tri_synthesis(const ft_triangle_fftw_plan * P, double * X, const int N, const int M);
void ft_execute_tri_analysis(const ft_triangle_fftw_plan * P, double * X, const int N, const int M);

typedef struct {
    fftw_plan planr1;
    fftw_plan planr2;
    fftw_plan planr3;
    fftw_plan planr4;
    fftw_plan plantheta;
    double * Y;
} ft_disk_fftw_plan;

void ft_destroy_disk_fftw_plan(ft_disk_fftw_plan * P);

ft_disk_fftw_plan * ft_plan_disk_with_kind(const int N, const int M, const fftw_r2r_kind kind[3][1]);
ft_disk_fftw_plan * ft_plan_disk_synthesis(const int N, const int M);
ft_disk_fftw_plan * ft_plan_disk_analysis(const int N, const int M);

void ft_execute_disk_synthesis(const ft_disk_fftw_plan * P, double * X, const int N, const int M);
void ft_execute_disk_analysis(const ft_disk_fftw_plan * P, double * X, const int N, const int M);


#endif //FASTTRANSFORMS_H
