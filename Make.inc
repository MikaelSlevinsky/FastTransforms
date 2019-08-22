LIB = fasttransforms
LIBDIR = .
LIBFLAGS = -shared -lm -fPIC

ifeq ($(OS), Windows_NT)
    UNAME := Windows
	SLIB = dll
else
	UNAME := $(shell uname)
	ifeq ($(UNAME), Darwin)
		SLIB = dylib
	else
		SLIB = so
	endif
endif

OBJ = src/transforms.c src/rotations.c src/permute.c src/tdc.c src/drivers.c src/fftw.c

#CC = gcc-8
#CC = /usr/local/opt/llvm/bin/clang

#BLAS = APPLEBLAS
#BLAS = OPENBLAS

#AVX512 = -mavx512f

CFLAGS = -std=gnu99 -Ofast $(AVX512) -march=native -mtune=native -mno-vzeroupper -I./src

ifeq ($(CC),/usr/local/opt/llvm/bin/clang)
    CFLAGS += -I/usr/local/opt/llvm/include
    LDFLAGS += -L/usr/local/opt/llvm/lib
endif

ifeq ($(BLAS), APPLEBLAS)
	CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
    LDFLAGS += -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
else ifeq ($(BLAS), OPENBLAS)
    ifeq ($(UNAME), Linux)
        CFLAGS += -I/usr/lib/openblas-base
        LDFLAGS += -L/usr/lib/openblas-base
    else ifeq ($(UNAME), Darwin)
        CFLAGS += -I/usr/local/opt/openblas/include
        LDFLAGS += -L/usr/local/opt/openblas/lib
        CFLAGS += -I/usr/local/opt/fftw/include
        LDFLAGS += -L/usr/local/opt/fftw/lib
    endif
endif
ifeq ($(UNAME), Windows)
    LDFLAGS += -L.
endif

LDLIBS += -lm -lquadmath -fopenmp -lblas -lfftw3
ifeq ($(UNAME), Linux)
    LDLIBS += -lfftw3_threads
else ifeq ($(UNAME), Darwin)
    LDLIBS += -lfftw3_threads
endif

FT_USE_MPFR=1

ifdef FT_USE_MPFR
    LDLIBS += -DFT_USE_MPFR -lmpfr
    ifeq ($(UNAME), Linux)
        LDLIBS += -lgmp
    else ifeq ($(UNAME), Darwin)
        LDLIBS += -lgmp
    else ifeq ($(OS), Windows_NT)
        LDLIBS += -lmpir
    endif
endif
