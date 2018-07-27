include Make.inc

all:
	make lib
	make assembly
	make examples
	make tests

lib:
	$(CC) $(CFLAGS) $(LIBFLAGS) $(OBJ) $(LDFLAGS) $(LDLIBS) -o lib$(LIB).$(SLIB)

assembly:
	$(CC) -S $(CFLAGS) test/test_assembly.c -o test_assembly.s

examples:
	$(CC) examples/additiontheorem.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o additiontheorem

tests:
	$(CC) test/utilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_transforms
	$(CC) test/utilities.c test/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_permute
	$(CC) test/utilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_rotations
	$(CC) test/utilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_drivers
	$(CC) test/utilities.c test/test_fftw.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_fftw

clean:
	rm -f lib$(LIB).$(SLIB)
	rm -f test_assembly.s
	rm -f additiontheorem
	rm -f test_transforms
	rm -f test_permute
	rm -f test_rotations
	rm -f test_drivers
	rm -f test_fftw

.PHONY: all lib assembly examples tests clean
