LIB = fasttransforms
LIBDIR = .
LIBFLAGS = -shared -fPIC

ifeq ($(FT_DETECT_GCC), 1)
    _GCCMAJOR := $(shell gfortran -dumpversion | cut -d'.' -f1)
    _GCCMINOR := $(shell gfortran -dumpversion | cut -d'.' -f2)
    ifeq ($(_GCCMAJOR),4)
      _GCCVER := $(_GCCMAJOR).$(_GCCMINOR)
    else
      _GCCVER := $(_GCCMAJOR)
    endif
    CC := gcc-$(_GCCVER)
    COV := gcov-$(_GCCVER)
endif

ifndef target
    target := $(shell $(CC) -dumpmachine)
endif

ifneq (, $(findstring darwin, $(target)))
    UNAME := Darwin
else ifneq (, $(findstring cygwin, $(target)))
    UNAME := Windows
else ifneq (, $(findstring mingw, $(target)))
    UNAME := Windows
else ifneq (, $(findstring msys, $(target)))
    UNAME := Windows
else ifneq (, $(findstring linux, $(target)))
    UNAME := Linux
else ifneq (, $(findstring freebsd, $(target)))
    UNAME := FreeBSD
else
    UNAME := Unknown
endif

simd := $(shell $(CC) -march=native -E -v - </dev/null 2>&1 | grep cc1)

ifneq (, $(filter -msse +sse, $(simd)))
    MSSE := -msse
endif
ifneq (, $(filter -msse2 +sse2, $(simd)))
    MSSE2 := -msse2
endif
ifneq (, $(filter -mavx +avx, $(simd)))
    MAVX := -mavx
endif
ifneq (, $(filter -mfma +fma, $(simd)))
    ifeq (, $(filter -mfma4 +fma4, $(simd)))
        MFMA := -mfma
    endif
endif
ifneq (, $(filter -mavx512f +avx512f -march=knl -march=knm -march=skylake-avx512 -march=cannonlake -march=icelake -march=cascadelake -march=cooperlake -march=tigerlake -march=sapphirerapids, $(simd)))
    ifneq ($(UNAME), Windows)
        MAVX512F := -mavx512f
    endif
endif

ifeq ($(UNAME), Darwin)
    SLIB = dylib
else ifeq ($(UNAME), Windows)
    SLIB = dll
else
    SLIB = so
endif

ASM = src/recurrence/recurrence_default.s \
      src/recurrence/recurrence_SSE.s \
      src/recurrence/recurrence_SSE2.s \
      src/recurrence/recurrence_AVX.s \
      src/recurrence/recurrence_AVX_FMA.s \
      src/recurrence/recurrence_AVX512F.s \
      src/recurrence/recurrence_NEON.s \
      src/permute/permute_default.s \
      src/permute/permute_SSE.s \
      src/permute/permute_SSE2.s \
      src/permute/permute_AVX.s \
      src/permute/permute_AVX512F.s \
      src/permute/permute_NEON.s \
      src/rotations/rotations_default.s \
      src/rotations/rotations_SSE2.s \
      src/rotations/rotations_AVX.s \
      src/rotations/rotations_AVX_FMA.s \
      src/rotations/rotations_AVX512F.s \
      src/rotations/rotations_NEON.s

SRC = src/recurrence.c src/transforms.c src/rotations.c src/permute.c src/tdc.c src/drivers.c src/fftw.c src/isometries.c src/helmholtzhodge.c

machine := $(shell $(CC) -dumpmachine | cut -d'-' -f1)

AFLAGS += -O3 -fPIC -std=gnu11 -I./src

ifndef CFLAGS
    CFLAGS = -O3
    ifneq (, $(findstring 86, $(machine)))
        CFLAGS += -march=native
    endif
    ifneq (, $(findstring aarch64, $(machine)))
        ifeq (, $(findstring clang, $(shell $(CC) --version)))
            CFLAGS += -mcpu=native
        endif
    endif
endif
CFLAGS += -std=gnu11 -I./src

ifeq ($(FT_COVERAGE), 1)
    CFLAGS += -coverage
endif

ifndef FT_NUM_THREADS
    FT_NUM_THREADS = 1
endif

ifdef FT_PREFIX
    CFLAGS += -I$(FT_PREFIX)/include
    ifeq ($(UNAME), Windows)
        LDFLAGS += -L$(FT_PREFIX)/bin
    else
        LDFLAGS += -L$(FT_PREFIX)/lib
    endif
else
    ifeq ($(UNAME), Darwin)
        CFLAGS += -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include
        LDFLAGS += -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
        ifeq ($(FT_USE_APPLEBLAS), 1)
            CFLAGS += -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
            LDFLAGS += -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
        else
            CFLAGS += -I/usr/local/opt/openblas/include
            LDFLAGS += -L/usr/local/opt/openblas/lib
        endif
        CFLAGS += -I/usr/local/opt/libomp/include
        LDFLAGS += -L/usr/local/opt/libomp/lib
        CFLAGS += -I/usr/local/opt/fftw/include
        LDFLAGS += -L/usr/local/opt/fftw/lib
        CFLAGS += -I/usr/local/opt/mpfr/include
        LDFLAGS += -L/usr/local/opt/mpfr/lib
        CFLAGS += -I/usr/local/opt/gmp/include
        LDFLAGS += -L/usr/local/opt/gmp/lib
    else ifeq ($(UNAME), Windows)
        CFLAGS += -Ic:\vcpkg\packages\openblas_x64-windows\include\openblas
        CFLAGS += -Ic:\vcpkg\packages\openblas_x64-windows\include
        CFLAGS += -Ic:\vcpkg\packages\fftw3_x64-windows\include
        CFLAGS += -Ic:\vcpkg\packages\mpfr_x64-windows\include
        LDFLAGS += -Lc:\vcpkg\packages\openblas_x64-windows\bin
        LDFLAGS += -Lc:\vcpkg\packages\fftw3_x64-windows\bin
        LDFLAGS += -Lc:\vcpkg\packages\mpfr_x64-windows\bin
    endif
endif

ifndef FT_BLAS
    FT_BLAS = blas
endif

ifndef FT_OPENMP
    ifneq (, $(findstring Apple LLVM version, $(shell $(CC) --version)))
        FT_OPENMP := -Xpreprocessor -fopenmp -lomp
    else ifneq (, $(findstring Apple clang version, $(shell $(CC) --version)))
        FT_OPENMP := -Xpreprocessor -fopenmp -lomp
    else ifeq ($(UNAME), FreeBSD)
        ifneq (, $(findstring clang, $(shell $(CC) --version)))
            FT_OPENMP := -fopenmp=libgomp
        else
            FT_OPENMP := -fopenmp
        endif
    else
        FT_OPENMP := -fopenmp
    endif
endif

LDLIBS += $(FT_OPENMP) -lm -lmpfr -l$(FT_BLAS) -lfftw3

ifdef FT_QUADMATH
    AFLAGS += -DFT_QUADMATH
    AFLAGS += -lquadmath
    CFLAGS += -DFT_QUADMATH
    LDLIBS += -lquadmath
endif

ifneq ($(FT_FFTW_WITH_COMBINED_THREADS), 1)
    LDLIBS += -lfftw3_threads
endif
