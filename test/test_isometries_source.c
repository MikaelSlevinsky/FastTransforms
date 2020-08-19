void Y(test_isometries)(int * checksum){
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("\t\t\t\t\t\t\t |   or Calculation Time\n");
    printf("---------------------------------------------------------|----------------------\n");

	int l = 50;
	int n = 2*l+1;
	int NTIMES = 3;
	struct timeval start, end;

	FLT alpha = 0.123;
	FLT beta = 0.456;
	FLT gamma = 0.789;

	FT_TIME({FLT * A = X(J)(l); free(A);}, start, end, NTIMES)
	printf("Time to build and free J using direct method \t\t |%20.6f s\n", elapsed(&start, &end, NTIMES));
	FT_TIME({FLT * A = X(J_eigen)(l); free(A);}, start, end, NTIMES)
	printf("Time to build and free J using eigenproblem \t\t |%20.6f s\n", elapsed(&start, &end, NTIMES));

	FT_TIME({FLT * A = X(rotation_matrix_direct)(l, alpha, beta, gamma); free(A);}, start, end, NTIMES)
	printf("Time to build and free rotation matrix, direct method \t |%20.6f s\n", elapsed(&start, &end, NTIMES));
	FT_TIME({FLT * A = X(rotation_matrix)(l, alpha, beta, gamma); free(A);}, start, end, NTIMES)
	printf("Time to build and free rotation matrix, eigenproblem \t |%20.6f s\n", elapsed(&start, &end, NTIMES));

	FLT * Jl = X(J_eigen)(l);
	FLT * Mult = (FLT*)calloc(n*n, sizeof(FLT));
	FLT * Id = (FLT*)calloc(n*n, sizeof(FLT));
	for(int i = 0; i < n; ++i)
		Id[i + i*n] = ONE(FLT);

	for(int j = 0; j < n; ++j){
		for(int i = 0; i < n; ++i){
			FLT entry = ZERO(FLT);
			for(int k = 0; k < n; ++k){
				entry += Jl[i + k*n] * Jl[k + j*n];
			}
			Mult[i + j*n] = entry;
		}
	}
	FLT err = X(norm_2arg)(Mult, Id, n*n)/X(norm_1arg)(Id, n);
    printf("Orthogonality of J matrix \t \t \t \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);
	free(Mult);

	Mult = (FLT*)calloc(n*n, sizeof(FLT));

	FLT * rot = X(rotation_matrix)(l, alpha, beta, gamma);
	FLT * rot_back = X(rotation_matrix)(l, -1*gamma, -1*beta, -1*alpha);
	
	for(int j = 0; j < n; ++j){
		for(int i = 0; i < n; ++i){
			FLT entry = ZERO(FLT);
			for(int k = 0; k < n; ++k){
				entry += rot[i + k*n] * rot_back[k + j*n];
			}
			Mult[i + j*n] = entry;
		}
	}
	err = X(norm_2arg)(Mult, Id, n*n)/X(norm_1arg)(Id, n);
    printf("Rotate by (alpha, beta, gamma), then rotate back. \t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);
}
