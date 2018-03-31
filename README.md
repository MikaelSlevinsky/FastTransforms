# FastTransforms

This library provides computational kernels and driver routines for two-dimensional harmonic polynomial transforms. The algorithms are backward stable with a runtime complexity of O(n<sup>3</sup>), where n is the polynomial degree, and are parallelized by OpenMP.

If you feel you need help getting started, please do not hesitate to e-mail me. If you use this library for your research, please cite the references.

## Performance Benchmark

The table below shows the current timings and accuracies for transforming spherical harmonic coefficients drawn from U(-1,1) to bivariate Fourier series and back. The timings are reported in seconds, executed on a Macbook Pro (Mid 2014) with a 2.8 GHz Intel Core i7 processor with 8 threads on 4 logical cores and 16 GB 1600 MHz DDR3 RAM. The error is measured in the relative 2- and ∞-norms, treating the matrices of coefficients as vectors.

| Degree            | 63       | 127      | 255      | 511      | 1023     | 2047     | 4095     |
| ----------------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| **Forward**       | 0.000088 | 0.000529 | 0.002194 | 0.017071 | 0.129575 | 1.171061 | 13.34318 |
| **Backward**      | 0.000092 | 0.000535 | 0.002409 | 0.017507 | 0.124157 | 0.898843 | 7.049812 |
| **ϵ<sub>2</sub>** | 6.01e-16 | 8.36e-16 | 9.68e-16 | 1.31e-15 | 1.82e-15 | 2.54e-15 | 3.55e-15 |
| **ϵ<sub>∞</sub>** | 1.55e-15 | 3.00e-15 | 3.22e-15 | 5.22e-15 | 8.44e-15 | 1.25e-14 | 1.71e-14 |

At the moment, the transforms are not the fastest in the world; however, they appear to be the most accurate.

The timings could be improved by saturating the vectorization capabilities and tuning transforms to specific architectures.

# References:

   [1]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, in press at *Appl. Comput. Harmon. Anal.*, 2017.

   [2]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.
