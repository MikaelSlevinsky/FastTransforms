include Make.inc

all:
	make lib
	make libf
	make assembly
	make tests
	make testsf

lib:
	$(CC) $(CFLAGS) $(LIBFLAGS) $(OBJ) $(LDFLAGS) $(LDLIBS) -o lib$(LIB).$(SLIB)

libf:
	$(CC) $(CFLAGS) $(LIBFLAGS) $(OBJF) $(LDFLAGS) $(LDLIBS) -o lib$(LIB)f.$(SLIB)

assembly:
	$(CC) -S $(CFLAGS) test/test_assembly.c -o test_assembly.s

tests:
	$(CC) test/double/utilities.c test/double/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_transforms
	$(CC) test/double/utilities.c test/double/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_permute
	$(CC) test/double/utilities.c test/double/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_rotations
	$(CC) test/double/utilities.c test/double/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB) $(LDFLAGS) $(LDLIBS) -o test_drivers

testsf:
	$(CC) test/float/utilitiesf.c test/float/test_transforms.c $(CFLAGS) -L$(LIBDIR) -l$(LIB)f $(LDFLAGS) $(LDLIBS) -o test_transformsf
	$(CC) test/float/utilitiesf.c test/float/test_permute.c $(CFLAGS) -L$(LIBDIR) -l$(LIB)f $(LDFLAGS) $(LDLIBS) -o test_permutef
	$(CC) test/float/utilitiesf.c test/float/test_rotations.c $(CFLAGS) -L$(LIBDIR) -l$(LIB)f $(LDFLAGS) $(LDLIBS) -o test_rotationsf
	$(CC) test/float/utilitiesf.c test/float/test_drivers.c $(CFLAGS) -L$(LIBDIR) -l$(LIB)f $(LDFLAGS) $(LDLIBS) -o test_driversf

clean:
	rm -f lib$(LIB).$(SLIB)
	rm -f lib$(LIB)f.$(SLIB)
	rm -f test_assembly.s
	rm -f test_transforms
	rm -f test_permute
	rm -f test_rotations
	rm -f test_drivers
	rm -f test_transformsf
	rm -f test_permutef
	rm -f test_rotationsf
	rm -f test_driversf
