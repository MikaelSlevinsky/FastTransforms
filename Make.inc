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

ifeq ($(UNAME), Darwin)
    SLIB = dylib
else ifeq ($(UNAME), Windows)
    SLIB = dll
else
    SLIB = so
endif

OBJ = src/transforms.c src/rotations.c src/permute.c src/tdc.c src/drivers.c src/fftw.c

machine := $(shell $(CC) -dumpmachine | cut -d'-' -f1)

ifndef CFLAGS
    ifneq (, $(findstring 86, $(machine)))
        CFLAGS = -march=native -mtune=native -mno-vzeroupper
    endif
endif
CFLAGS += -Ofast -std=gnu99 -I./src

ifdef FT_PREFIX
    CFLAGS += -I$(FT_PREFIX)/include
    ifeq ($(UNAME), Windows)
        LDFLAGS += -L$(FT_PREFIX)/bin
    else
        LDFLAGS += -L$(FT_PREFIX)/lib
    endif
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
        CFLAGS += -Ic:\tools\vcpkg\installed\x64-windows\include\openblas
        CFLAGS += -Ic:\tools\vcpkg\installed\x64-windows\include
        LDFLAGS += -Lc:\tools\vcpkg\installed\x64-windows\bin
    endif
endif

ifndef FT_BLAS
    FT_BLAS = blas
endif

LDLIBS += -fopenmp -lm -lquadmath -lmpfr -l$(FT_BLAS) -lfftw3

ifneq ($(FT_FFTW_WITH_COMBINED_THREADS), 1)
    LDLIBS += -lfftw3_threads
endif
