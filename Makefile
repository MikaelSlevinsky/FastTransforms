FTHOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(FTHOME)/Make.inc

ifeq ($(OS), Windows_NT)
	SLIB = dll
else
	UNAME := $(shell uname)
	ifeq ($(UNAME), Darwin)
		SLIB = dylib
	else
		SLIB = so
	endif
endif

OBJ = src/transforms.c src/rotations.c src/drivers.c src/permute.c
CFLAGS = -Ofast -march=native -mtune=native -I./src
LIBFLAGS = -shared -fPIC -lm -lgomp -fopenmp
LIBDIR = .
LIB = fasttransforms

ifeq ($(USE_SYSTEM_BLAS),0)
	CFLAGS += -I/Applications/julia/deps/openblas -L/Applications/julia/deps/openblas -lopenblas
else
	CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers -lblas
endif

all:
	make lib
	make tests

lib:
	gcc-8 $(CFLAGS) $(LIBFLAGS) $(OBJ) -o lib$(LIB).$(SLIB)

assembly:
	gcc-8 -S $(CFLAGS) -mavx512f test/test_assembly.c -o test_assembly.s

tests:
	make test_transforms
	make test_permute
	make test_rotations
	make test_drivers

test_transforms:
	gcc-8 test/utilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_transforms

test_permute:
	gcc-8 test/utilities.c test/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_permute

test_rotations:
	gcc-8 test/utilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_rotations

test_drivers:
	gcc-8 test/utilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_drivers

clean:
	rm -f lib$(LIB).$(SLIB)
	rm -f test_transforms
	rm -f test_permute
	rm -f test_rotations
	rm -f test_drivers
