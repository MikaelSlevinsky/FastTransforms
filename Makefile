include Make.inc

all:
	make assembly
	make lib
	make tests
	make examples

assembly:
	$(CC) -S $(CFLAGS) test/test_assembly.c -o test_assembly.s

lib:
	$(CC) $(CFLAGS) $(LIBFLAGS) $(OBJ) $(LDFLAGS) $(LDLIBS) -o lib$(LIB).$(SLIB)

tests:
	$(CC) src/ftutilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_transforms
	$(CC) src/ftutilities.c test/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_permute
	$(CC) src/ftutilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_rotations
	$(CC) src/ftutilities.c test/test_arrow.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_arrow
	$(CC) src/ftutilities.c test/test_tridiagonal.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_tridiagonal
	$(CC) src/ftutilities.c test/test_hierarchical.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_hierarchical
	$(CC) src/ftutilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_drivers
	$(CC) src/ftutilities.c test/test_fftw.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_fftw

examples:
	$(CC) src/ftutilities.c examples/additiontheorem.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o additiontheorem
	$(CC) src/ftutilities.c examples/holomorphic.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o holomorphic

clean:
	rm -f lib$(LIB).$(SLIB)
	rm -f additiontheorem
	rm -f holomorphic
	rm -f test_*

.PHONY: all lib assembly examples tests clean
