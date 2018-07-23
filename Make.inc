LIB = fasttransforms
LIBDIR = .
LIBFLAGS = -shared -fPIC -fopenmp

ifeq ($(OS), Windows_NT)
	SLIB = dll
else
	UNAME = $(shell uname)
	ifeq ($(UNAME), Darwin)
		SLIB = dylib
	else
		SLIB = so
	endif
endif

OBJ = src/double/transforms.c src/double/rotations.c src/double/permute.c src/double/drivers.c
OBJF = src/float/transforms.c src/float/rotations.c src/float/permute.c src/float/drivers.c

CC = gcc-8
#CC = /usr/local/opt/llvm/bin/clang

AVX512 =# -mavx512f
BLAS = APPLEBLAS
#BLAS = OPENBLAS

CFLAGS = -Ofast $(AVX512) -march=native -mtune=native -I./src

ifeq ($(CC),/usr/local/opt/llvm/bin/clang)
	CFLAGS += -I/usr/local/opt/llvm/include
    LDFLAGS += -L/usr/local/opt/llvm/lib
endif

ifeq ($(BLAS),APPLEBLAS)
	CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
else
	CFLAGS += -I/Applications/julia/deps/openblas
    LDFLAGS += -L/Applications/julia/deps/openblas
endif
LDLIBS = -lblas
