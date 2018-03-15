# FastTransforms

This library provides computational kernels and driver routines for two-
dimensional harmonic polynomial transforms. The algorithms are backward stable
with a runtime complexity of O(n<sup>3</sup>), where n is the polynomial
degree, and are parallelized by OpenMP.

If you use this library for your research, please cite the references.

# References:

   [1]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, in press at *Appl. Comput. Harmon. Anal.*, 2017.

   [2]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.
