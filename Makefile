include Make.inc

all:
	make assembly
	make lib
	make tests
	make examples

win:
	make assembly
	make lib
	make wintests
	make examples

assembly:
	$(CC) -S $(CFLAGS) test/test_assembly.c -o test_assembly.s

lib:
	$(CC) $(CFLAGS) $(LIBFLAGS) $(OBJ) $(LDFLAGS) $(LDLIBS) -o lib$(LIB).$(SLIB)

tests:
	$(CC) src/ftutilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_transforms
	$(CC) src/ftutilities.c test/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_permute
	$(CC) src/ftutilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_rotations
	$(CC) src/ftutilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_drivers
	$(CC) src/ftutilities.c test/test_fftw.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_fftw

wintests:
	$(CC) src/ftutilities.c test/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_transforms
	$(CC) src/ftutilities.c test/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_permute
	$(CC) src/ftutilities.c test/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_rotations
	$(CC) src/ftutilities.c test/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_drivers

examples:
	$(CC) src/ftutilities.c examples/additiontheorem.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o additiontheorem
	$(CC) src/ftutilities.c examples/holomorphic.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o holomorphic

clean:
	rm -f lib$(LIB).$(SLIB)
	rm -f test_assembly.s
	rm -f additiontheorem
	rm -f holomorphic
	rm -f test_transforms
	rm -f test_permute
	rm -f test_rotations
	rm -f test_drivers
	rm -f test_fftw

.PHONY: all lib assembly examples tests clean
