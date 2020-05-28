# FastTransforms

[![Travis](https://travis-ci.org/MikaelSlevinsky/FastTransforms.svg?branch=master)](https://travis-ci.org/MikaelSlevinsky/FastTransforms) [![Build status](https://ci.appveyor.com/api/projects/status/er98t0q3bsx4a5l9/branch/master?svg=true)](https://ci.appveyor.com/project/MikaelSlevinsky/fasttransforms/branch/master) [![codecov](https://codecov.io/gh/MikaelSlevinsky/FastTransforms/branch/master/graph/badge.svg)](https://codecov.io/gh/MikaelSlevinsky/FastTransforms) [![](https://img.shields.io/badge/docs-master-blue.svg)](https://MikaelSlevinsky.github.io/FastTransforms)

[FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms) provides computational kernels and driver routines for orthogonal polynomial transforms. The univariate algorithms have a runtime complexity of O(n log n), while the multivariate algorithms are 2-normwise backward stable with a runtime complexity of O(n<sup>d+1</sup>), where n is the polynomial degree and d is the spatial dimension of the problem.

The transforms are separated into computational kernels that offer SSE, AVX, and AVX-512 vectorization on applicable Intel processors, and driver routines that are easily parallelized by OpenMP.

If you feel you need help getting started, please do not hesitate to e-mail me. If you use this library for your research, please cite the references.

## Installation Notes

The library makes use of OpenBLAS, FFTW3, and MPFR, which are easily installed via package managers such as Homebrew, apt-get, or vcpkg. When `FastTransforms` is compiled with OpenMP, the environment variable that controls multithreading is `OMP_NUM_THREADS`.

### macOS

Apple's version of GCC does not support OpenMP. Sample installation:
```
brew install gcc@8 fftw mpfr
make CC=gcc-8 FT_USE_APPLEBLAS=1
```
On macOS, the OpenBLAS dependency is optional in light of the vecLib framework. In case the library is compiled with vecLib, then the environment variable `VECLIB_MAXIMUM_THREADS` partially controls the multithreading.

### Linux

To access functions from the library, you must ensure that you append the current library path to the default. Sample installation:
```
apt-get install gcc-8 libblas-dev libopenblas-base libfftw3-dev libmpfr-dev
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
make CC=gcc-8
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

The library is compiled by GNU GCC 8.3.0 as above with the compiler optimization flags `-O3 -march=native`, and executed on an iMac Pro (Early 2018) with a 2.3 GHz Intel Xeon W-2191B processor with 18 threads on 18 logical cores.

### Spherical harmonic series to bivariate Fourier series (`sph2fourier`)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000159 | 0.000540 | 0.001852 | 0.005510 | 0.022874 | 0.120082 | 0.786019 | 4.902573 |
| **Backward**      | 0.000159 | 0.000555 | 0.001852 | 0.005501 | 0.023940 | 0.119339 | 0.778569 | 4.939324 |
| **ϵ<sub>2</sub>** | 5.42e-16 | 7.79e-16 | 9.23e-16 | 1.27e-15 | 1.80e-15 | 2.52e-15 | 3.54e-15 | 4.98e-15 |
| **ϵ<sub>∞</sub>** | 1.33e-15 | 2.55e-15 | 4.55e-15 | 5.22e-15 | 9.33e-15 | 1.11e-14 | 1.81e-14 | 3.80e-14 |

### Proriol series to bivariate Chebyshev series (`tri2cheb`) with (α, β, γ) = (0, 0, 0)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000128 | 0.000313 | 0.000969 | 0.003482 | 0.015711 | 0.093525 | 0.638155 | 4.591086 |
| **Backward**      | 0.000125 | 0.000302 | 0.000948 | 0.003543 | 0.016041 | 0.094016 | 0.670728 | 4.420676 |
| **ϵ<sub>2</sub>** | 2.30e-15 | 3.94e-15 | 8.14e-15 | 1.68e-14 | 3.55e-14 | 7.30e-14 | 1.44e-13 | 2.97e-13 |
| **ϵ<sub>∞</sub>** | 1.09e-14 | 1.84e-14 | 5.49e-14 | 1.49e-13 | 9.06e-13 | 1.95e-12 | 3.73e-12 | 1.05e-11 |

The error growth rate is significantly reduced if (α, β, γ) = (-0.5, -0.5, -0.5).

### Zernike series to Chebyshev×Fourier series (`disk2cxf`)

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000275 | 0.001069 | 0.003448 | 0.011251 | 0.053583 | 0.329135 | 2.157533 | 15.33666 |
| **Backward**      | 0.000280 | 0.001031 | 0.003435 | 0.011320 | 0.053093 | 0.317026 | 2.203625 | 15.42320 |
| **ϵ<sub>2</sub>** | 7.60e-16 | 9.35e-16 | 1.31e-15 | 1.84e-15 | 2.56e-15 | 3.59e-15 | 5.05e-15 | 7.15e-15 |
| **ϵ<sub>∞</sub>** | 1.78e-15 | 2.66e-15 | 5.33e-15 | 7.11e-15 | 1.27e-14 | 1.79e-14 | 2.91e-14 | 1.66e-13 |

# References:

   [1]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, *Appl. Comput. Harmon. Anal.*, **47**:585—606, 2019.

   [2]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.
