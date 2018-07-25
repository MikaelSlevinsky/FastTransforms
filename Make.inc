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

OBJ = src/transforms.c src/rotations.c src/permute.c src/drivers.c

#CC = gcc-8
#CC = /usr/local/opt/llvm/bin/clang

AVX512 =# -mavx512f
#BLAS = APPLEBLAS
BLAS = OPENBLAS

CFLAGS = -Ofast $(AVX512) -march=native -mtune=native -I./src

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
    endif
endif
LDLIBS += -lblas
