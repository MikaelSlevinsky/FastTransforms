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

ifeq ($(FT_DETECT_GCC), 1)
    _GCCMAJOR := $(shell gfortran -dumpversion | cut -d'.' -f1)
    _GCCMINOR := $(shell gfortran -dumpversion | cut -d'.' -f2)
    ifeq ($(_GCCMAJOR),4)
      _GCCVER := $(_GCCMAJOR).$(_GCCMINOR)
    else
      _GCCVER := $(_GCCMAJOR)
    endif
    CC := gcc-$(_GCCVER)
endif

CFLAGS += -std=gnu99 -Ofast -march=native -mtune=native -mno-vzeroupper -I./src

ifeq ($(FT_USE_PREDEFINED_LIBRARIES), 1)
    CFLAGS += -I$(prefix)/include
    LDFLAGS += -L$(prefix)/lib
else
    ifeq ($(UNAME), Darwin)
        ifeq ($(FT_USE_APPLEBLAS), 1)
            CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
            LDFLAGS += -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
        else
            CFLAGS += -I/usr/local/opt/openblas/include
            LDFLAGS += -L/usr/local/opt/openblas/lib
        endif
        CFLAGS += -I/usr/local/opt/fftw/include
        LDFLAGS += -L/usr/local/opt/fftw/lib
        CFLAGS += -I/usr/local/opt/mpfr/include
        LDFLAGS += -L/usr/local/opt/mpfr/lib
        CFLAGS += -I/usr/local/opt/gmp/include
        LDFLAGS += -L/usr/local/opt/gmp/lib
    else ifeq ($(UNAME), Windows)
        LDFLAGS += -L.
    endif
endif

LDLIBS += -lm -lquadmath -fopenmp -lblas -lfftw3

ifneq ($(FT_FFTW_WITH_COMBINED_THREADS), 1)
    LDLIBS += -lfftw3_threads
endif

LDLIBS += -lmpfr
ifeq ($(UNAME), Linux)
    LDLIBS += -lgmp
else ifeq ($(UNAME), Darwin)
    LDLIBS += -lgmp
else ifeq ($(OS), Windows_NT)
    LDLIBS += -lmpir
endif
