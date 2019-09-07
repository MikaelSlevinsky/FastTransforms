# FastTransforms

[![Travis](https://travis-ci.org/MikaelSlevinsky/FastTransforms.svg?branch=master)](https://travis-ci.org/MikaelSlevinsky/FastTransforms) [![Build status](https://ci.appveyor.com/api/projects/status/er98t0q3bsx4a5l9/branch/master?svg=true)](https://ci.appveyor.com/project/MikaelSlevinsky/fasttransforms/branch/master) [![](https://img.shields.io/badge/docs-master-blue.svg)](https://MikaelSlevinsky.github.io/FastTransforms)

[FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms) provides computational kernels and driver routines for orthogonal polynomial transforms. The univariate algorithms have a runtime complexity of O(n log n), while the multivariate algorithms are 2-normwise backward stable with a runtime complexity of O(n<sup>d+1</sup>), where n is the polynomial degree and d is the spatial dimension of the problem.

The transforms are separated into computational kernels that offer SSE, AVX, and AVX-512 vectorization on applicable Intel processors, and driver routines that are easily parallelized by OpenMP.

If you feel you need help getting started, please do not hesitate to e-mail me. If you use this library for your research, please cite the references.

## Installation Notes

The library makes use of OpenBLAS, FFTW3, and MPFR, which are easily installed via package managers such as Homebrew, apt-get, or vcpkg. When `FastTransforms` is compiled with OpenMP, the environment variable that controls multithreading is `OMP_NUM_THREADS`.

### macOS

Apple's version of GCC does not support OpenMP. Sample installation:
```
brew install gcc@8 fftw mpfr
export CC=gcc-8 && export FT_USE_APPLEBLAS=1 && make
```
On macOS, the OpenBLAS dependency is optional in light of the vecLib framework. In case the library is compiled with vecLib, then the environment variable `VECLIB_MAXIMUM_THREADS` partially controls the multithreading.

### Linux

To access functions from the library, you must ensure that you append the current library path to the default. Sample installation:
```
apt-get install gcc-8 libblas-dev libopenblas-base libfftw3-dev libmpfr-dev
export CC=gcc-8 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. && make
```
See the [Travis build](https://github.com/MikaelSlevinsky/FastTransforms/blob/master/.travis.yml) for further details.

### Windows

We use GCC 7.2.0 distributed through MinGW-w64 on Visual Studio 2015 and 2017. Sample installation:
```
vcpkg install openblas:x64-windows fftw3:x64-windows mpir:x64-windows mpfr:x64-windows
set CC=gcc && mingw32-make
```
See the [AppVeyor build](https://github.com/MikaelSlevinsky/FastTransforms/blob/master/.appveyor.yml) for further details.

## Performance Benchmark

The tables below shows the current timings and accuracies for transforming orthonormal spherical harmonic coefficients drawn from U(-1,1) to bivariate Fourier series and back. The timings are reported in seconds and the error is measured in the relative 2- and ∞-norms, treating the matrices of coefficients as vectors.

### MacBook Pro

The library is compiled by GNU GCC 7.3.0 with the flags `-Ofast -march=native -mtune=native`, and executed on a MacBook Pro (Mid 2014) with a 2.8 GHz Intel Core i7-4980HQ processor with 8 threads on 4 logical cores.

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000088 | 0.000529 | 0.002194 | 0.017071 | 0.121610 | 1.033182 | 12.38314 | 139.3769 |
| **Backward**      | 0.000092 | 0.000535 | 0.002409 | 0.017507 | 0.111388 | 0.795846 | 6.526968 | 101.6704 |
| **ϵ<sub>2</sub>** | 6.01e-16 | 8.36e-16 | 9.68e-16 | 1.31e-15 | 1.82e-15 | 2.54e-15 | 3.55e-15 | 5.00e-15 |
| **ϵ<sub>∞</sub>** | 1.55e-15 | 3.00e-15 | 3.22e-15 | 5.22e-15 | 8.44e-15 | 1.25e-14 | 1.71e-14 | 3.62e-14 |

### iMac Pro

The library is compiled by LLVM Clang-6.0.0 with the flags `-Ofast -march=native -mtune=native`, and executed on an iMac Pro (Early 2018) with a 2.3 GHz Intel Xeon W-2191B processor with 18 threads on 18 logical cores.

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000091 | 0.000513 | 0.001119 | 0.004180 | 0.021416 | 0.185732 | 2.445858 | 24.53793 |
| **Backward**      | 0.000089 | 0.000516 | 0.001105 | 0.004265 | 0.022442 | 0.130701 | 1.386275 | 20.40340 |
| **ϵ<sub>2</sub>** | 6.69e-16 | 8.55e-16 | 1.05e-15 | 1.41e-15 | 1.95e-15 | 2.79e-15 | 3.95e-15 | 5.86e-15 |
| **ϵ<sub>∞</sub>** | 2.44e-15 | 2.78e-15 | 4.44e-15 | 6.33e-15 | 8.22e-15 | 1.23e-14 | 1.99e-14 | 4.17e-14 |


# References:

   [1]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, *Appl. Comput. Harmon. Anal.*, **47**:585—606, 2019.

   [2]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.
