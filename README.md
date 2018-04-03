# FastTransforms

This library provides computational kernels and driver routines for two-dimensional harmonic polynomial transforms. The algorithms are backward stable with a runtime complexity of O(n<sup>3</sup>), where n is the polynomial degree, and are parallelized by OpenMP.

If you feel you need help getting started, please do not hesitate to e-mail me. If you use this library for your research, please cite the references.

## Performance Benchmark

The table below shows the current timings and accuracies for transforming orthonormal spherical harmonic coefficients drawn from U(-1,1) to bivariate Fourier series and back. The timings are reported in seconds, compiled by GNU GCC 7.3.0 with the flags `-Ofast -march=native`, and executed on a MacBook Pro (Mid 2014) with a 2.8 GHz Intel Core i7 processor with 8 threads on 4 logical cores, 1 MB of L2 cache, 6 MB of L3 cache, and 16 GB 1600 MHz DDR3 RAM. The error is measured in the relative 2- and ∞-norms, treating the matrices of coefficients as vectors.

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     | 8191     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000088 | 0.000529 | 0.002194 | 0.017071 | 0.121610 | 1.033182 | 12.38314 | 139.3769 |
| **Backward**      | 0.000092 | 0.000535 | 0.002409 | 0.017507 | 0.111388 | 0.795846 | 6.526968 | 101.6704 |
| **ϵ<sub>2</sub>** | 6.01e-16 | 8.36e-16 | 9.68e-16 | 1.31e-15 | 1.82e-15 | 2.54e-15 | 3.55e-15 | 5.00e-15 |
| **ϵ<sub>∞</sub>** | 1.55e-15 | 3.00e-15 | 3.22e-15 | 5.22e-15 | 8.44e-15 | 1.25e-14 | 1.71e-14 | 3.62e-14 |

At the moment, the transforms are not the fastest in the world; however, they appear to be the most accurate.

The timings could be improved by saturating the vectorization capabilities and tuning transforms to specific architectures.

# References:

   [1]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, in press at *Appl. Comput. Harmon. Anal.*, 2017.

   [2]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.
