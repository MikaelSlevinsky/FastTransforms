include Make.inc

all:
	make assembly
	make lib
	make tests
	make examples
	make rotation_sph

assembly:
	$(CC) -S $(AFLAGS) src/recurrence/recurrence_default.c -o src/recurrence/recurrence_default.s
	$(CC) -S $(AFLAGS) $(MSSE) src/recurrence/recurrence_SSE.c -o src/recurrence/recurrence_SSE.s
	$(CC) -S $(AFLAGS) $(MSSE2) src/recurrence/recurrence_SSE2.c -o src/recurrence/recurrence_SSE2.s
	$(CC) -S $(AFLAGS) $(MAVX) src/recurrence/recurrence_AVX.c -o src/recurrence/recurrence_AVX.s
	$(CC) -S $(AFLAGS) $(MAVX) $(MFMA) src/recurrence/recurrence_AVX_FMA.c -o src/recurrence/recurrence_AVX_FMA.s
	$(CC) -S $(AFLAGS) $(MAVX512F) src/recurrence/recurrence_AVX512F.c -o src/recurrence/recurrence_AVX512F.s

	$(CC) -S $(AFLAGS) src/permute/permute_default.c -o src/permute/permute_default.s
	$(CC) -S $(AFLAGS) $(MSSE) src/permute/permute_SSE.c -o src/permute/permute_SSE.s
	$(CC) -S $(AFLAGS) $(MSSE2) src/permute/permute_SSE2.c -o src/permute/permute_SSE2.s
	$(CC) -S $(AFLAGS) $(MAVX) src/permute/permute_AVX.c -o src/permute/permute_AVX.s
	$(CC) -S $(AFLAGS) $(MAVX512F) src/permute/permute_AVX512F.c -o src/permute/permute_AVX512F.s

	$(CC) -S $(AFLAGS) src/rotations/rotations_default.c -o src/rotations/rotations_default.s
	$(CC) -S $(AFLAGS) $(MSSE2) src/rotations/rotations_SSE2.c -o src/rotations/rotations_SSE2.s
	$(CC) -S $(AFLAGS) $(MAVX) src/rotations/rotations_AVX.c -o src/rotations/rotations_AVX.s
	$(CC) -S $(AFLAGS) $(MAVX) $(MFMA) src/rotations/rotations_AVX_FMA.c -o src/rotations/rotations_AVX_FMA.s
	$(CC) -S $(AFLAGS) $(MAVX512F) src/rotations/rotations_AVX512F.c -o src/rotations/rotations_AVX512F.s

lib:
	$(CC) $(CFLAGS) $(LIBFLAGS) $(ASM) $(SRC) $(LDFLAGS) $(LDLIBS) -o lib$(LIB).$(SLIB)

tests:
	$(CC) src/ftutilities.c test/test_recurrence.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_recurrence
	$(CC) src/ftutilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_transforms
	$(CC) src/ftutilities.c test/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_permute
	$(CC) src/ftutilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_rotations
	$(CC) src/ftutilities.c test/test_tridiagonal.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_tridiagonal
	$(CC) src/ftutilities.c test/test_hierarchical.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_hierarchical
	$(CC) src/ftutilities.c test/test_banded.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_banded
	$(CC) src/ftutilities.c test/test_dprk.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_dprk
	$(CC) src/ftutilities.c test/test_tdc.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_tdc
	$(CC) src/ftutilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_drivers
	$(CC) src/ftutilities.c test/test_fftw.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_fftw

examples:
	$(CC) src/ftutilities.c examples/additiontheorem.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o additiontheorem
	$(CC) src/ftutilities.c examples/calculus.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o calculus
	$(CC) src/ftutilities.c examples/holomorphic.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o holomorphic
	$(CC) src/ftutilities.c examples/nonlocaldiffusion.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o nonlocaldiffusion
	$(CC) src/ftutilities.c examples/spinweighted.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o spinweighted

rotation_sph: rotation_sph.c rotation_sph_source.c
	$(CC) src/ftutilities.c rotation_sph.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) -lopenblas $(LDFLAGS) $(LDLIBS) -o rotation_sph

clean:
	rm -f lib$(LIB).*
	rm -f src/recurrence/*.s
	rm -f src/permute/*.s
	rm -f src/rotations/*.s
	rm -f additiontheorem
	rm -f calculus
	rm -f holomorphic
	rm -f nonlocaldiffusion
	rm -f spinweighted
	rm -f test_*

.PHONY: all assembly lib tests examples clean
