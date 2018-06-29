export VECLIB_MAXIMUM_THREADS=18
export OMP_NUM_THREADS=18
USE_SYSTEM_BLAS := 1

CC = /usr/local/opt/llvm/bin/clang
AVX512 = -mavx512f
