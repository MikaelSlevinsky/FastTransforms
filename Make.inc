LIB = fasttransforms
LIBDIR = .
LIBFLAGS = -shared -lm -fPIC -fopenmp

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

OBJ = src/transforms.c src/rotations.c src/permute.c src/drivers.c src/fftw.c

#CC = gcc-8
#CC = /usr/local/opt/llvm/bin/clang

#BLAS = APPLEBLAS
#BLAS = OPENBLAS

<<<<<<< HEAD
AVX512 =# -mavx512f 
=======
#AVX512 = -mavx512f
>>>>>>> master

CFLAGS = -std=gnu99 -Ofast $(AVX512) -march=native -mtune=native -I./src

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
LDLIBS += -lm -lblas -lfftw3 -lfftw3_threads
