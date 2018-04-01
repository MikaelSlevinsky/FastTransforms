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

OBJ = src/transforms.c src/rotations.c src/drivers.c
CFLAGS = -Ofast -march=native
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
	gcc-7 $(CFLAGS) $(LIBFLAGS) $(OBJ) -o lib$(LIB).$(SLIB)

tests:
	make test_transforms
	make test_rotations
	make test_drivers

test_transforms:
	gcc-7 test/utilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_transforms

test_rotations:
	gcc-7 test/utilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_rotations

test_drivers:
	gcc-7 test/utilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -o test_drivers

clean:
	rm lib$(LIB).$(SLIB)
	rm test_transforms
	rm test_rotations
	rm test_drivers
