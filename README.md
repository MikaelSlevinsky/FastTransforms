# FastTransforms

[![Travis](https://travis-ci.com/MikaelSlevinsky/FastTransforms.svg?branch=master)](https://travis-ci.com/MikaelSlevinsky/FastTransforms) [![Build status](https://ci.appveyor.com/api/projects/status/er98t0q3bsx4a5l9/branch/master?svg=true)](https://ci.appveyor.com/project/MikaelSlevinsky/fasttransforms/branch/master) [![codecov](https://codecov.io/gh/MikaelSlevinsky/FastTransforms/branch/master/graph/badge.svg)](https://codecov.io/gh/MikaelSlevinsky/FastTransforms) [![](https://img.shields.io/badge/docs-master-blue.svg)](https://MikaelSlevinsky.github.io/FastTransforms)

[FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms) provides computational kernels and driver routines for orthogonal polynomial transforms. The univariate algorithms have a runtime complexity of O(n log n), while the multivariate algorithms are 2-normwise backward stable with a runtime complexity of O(n<sup>d+1</sup>), where n is the polynomial degree and d is the spatial dimension of the problem.

The transforms are separated into computational kernels that offer SSE, AVX, and AVX-512 vectorization on applicable Intel processors, and driver routines that are easily parallelized by OpenMP.

If you feel you need help getting started, please do not hesitate to e-mail me. If you use this library for your research, please cite the references.

## Installation Notes

The library makes use of OpenBLAS, FFTW3, and MPFR, which are easily installed via package managers such as Homebrew, apt-get, or vcpkg. When `FastTransforms` is compiled with OpenMP, the environment variable that controls multithreading is `OMP_NUM_THREADS`.

### macOS

Sample installation:
```
brew install libomp fftw mpfr
make CC=clang FT_USE_APPLEBLAS=1
```
On macOS, the OpenBLAS dependency is optional in light of the vecLib framework. It is also important to have `export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"` defined in your environment. In case the library is compiled with vecLib, then the environment variable `VECLIB_MAXIMUM_THREADS` partially controls the multithreading.

### Linux

To access functions from the library, you must ensure that you append the current library path to the default. Sample installation:
```
apt-get install libomp-dev libblas-dev libopenblas-base libfftw3-dev libmpfr-dev
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
make CC=gcc
```
See the [Travis build](https://github.com/MikaelSlevinsky/FastTransforms/blob/master/.travis.yml) for further details.

### Windows

We use GCC 7.2.0 distributed through MinGW-w64 on Visual Studio 2017. Sample installation:
```
vcpkg install openblas:x64-windows fftw3[core,threads]:x64-windows mpir:x64-windows mpfr:x64-windows
mingw32-make CC=gcc FT_BLAS=openblas FT_FFTW_WITH_COMBINED_THREADS=1
```
See the [AppVeyor build](https://github.com/MikaelSlevinsky/FastTransforms/blob/master/.appveyor.yml) for further details.

## Performance Benchmark

The tables below shows the current timings and accuracies for the transforms with pseudorandom coefficients drawn from U(-1,1) and converted back. The timings are reported in seconds and the error is measured in the relative 2- and ∞-norms.

The library is compiled by Apple LLVM version 10.0.1 as above with the compiler optimization flags `-O3 -march=native`, and executed on an iMac Pro (Early 2018) with a 2.3 GHz Intel Xeon W-2191B processor with 18 threads on 18 logical cores.

### Spherical harmonic series to bivariate Fourier series (`sph2fourier`)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000101 | 0.000446 | 0.001952 | 0.006277 | 0.026590 | 0.141962 | 0.775724 | 4.870621 |
| **Backward**      | 0.000101 | 0.000448 | 0.001918 | 0.006357 | 0.027123 | 0.145149 | 0.815881 | 5.040577 |
| **ϵ<sub>2</sub>** | 5.42e-16 | 7.79e-16 | 9.23e-16 | 1.27e-15 | 1.80e-15 | 2.52e-15 | 3.54e-15 | 4.98e-15 |
| **ϵ<sub>∞</sub>** | 1.33e-15 | 2.55e-15 | 4.55e-15 | 5.22e-15 | 9.33e-15 | 1.11e-14 | 1.81e-14 | 3.80e-14 |

### Proriol series to bivariate Chebyshev series (`tri2cheb`) with (α, β, γ) = (0, 0, 0)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000063 | 0.000300 | 0.001048 | 0.003886 | 0.019123 | 0.113299 | 0.699465 | 4.555086 |
| **Backward**      | 0.000065 | 0.000289 | 0.000987 | 0.003837 | 0.019529 | 0.114718 | 0.699051 | 4.697958 |
| **ϵ<sub>2</sub>** | 2.30e-15 | 3.94e-15 | 8.14e-15 | 1.68e-14 | 3.55e-14 | 7.30e-14 | 1.44e-13 | 2.97e-13 |
| **ϵ<sub>∞</sub>** | 1.09e-14 | 1.84e-14 | 5.49e-14 | 1.49e-13 | 9.06e-13 | 1.95e-12 | 3.73e-12 | 1.05e-11 |

The error growth rate is significantly reduced if (α, β, γ) = (-0.5, -0.5, -0.5).

### Generalized Zernike series to Chebyshev×Fourier series (`disk2cxf`) with (α, β) = (0, 0)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000209 | 0.001077 | 0.003618 | 0.013243 | 0.064321 | 0.369272 | 2.182001 | 16.15872 |
| **Backward**      | 0.000208 | 0.001027 | 0.003487 | 0.012903 | 0.064544 | 0.372493 | 2.191747 | 16.43340 |
| **ϵ<sub>2</sub>** | 7.60e-16 | 9.35e-16 | 1.31e-15 | 1.84e-15 | 2.56e-15 | 3.59e-15 | 5.05e-15 | 7.15e-15 |
| **ϵ<sub>∞</sub>** | 1.78e-15 | 2.66e-15 | 5.33e-15 | 7.11e-15 | 1.27e-14 | 1.79e-14 | 2.91e-14 | 1.66e-13 |

### Rotation of a spherical harmonic series (`sph_rotation`) with ZYZ Euler angles (α, β, γ) = (0.1, 0.2, 0.3)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Time**          | 0.000138 | 0.000382 | 0.001290 | 0.005414 | 0.026758 | 0.156653 | 1.105567 | 9.460644 |
| **ϵ<sub>2</sub>** | 3.34e-14 | 8.22e-14 | 2.24e-13 | 5.56e-13 | 1.50e-12 | 4.23e-12 | 1.18e-11 | 3.30e-11 |
| **ϵ<sub>∞</sub>** | 2.41e-13 | 1.51e-12 | 5.86e-12 | 2.29e-11 | 8.04e-11 | 3.02e-10 | 1.38e-09 | 5.67e-09 |

# References:

   [1] J. Molina and R. M. Slevinsky. <a href="https://arxiv.org/abs/1809.04555">A rapid and well-conditioned algorithm for the Helmholtz–Hodge decomposition of vector fields on the sphere</a>, arXiv:1809.04555, 2018.

   [1] S. Olver, R. M. Slevinsky, and A. Townsend. <a href="https://doi.org/10.1017/S0962492920000045">Fast algorithms using orthogonal polynomials</a>, *Acta Numerica*, **29**:573—699, 2020.

   [2]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, *Appl. Comput. Harmon. Anal.*, **47**:585—606, 2019.

   [3]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.
