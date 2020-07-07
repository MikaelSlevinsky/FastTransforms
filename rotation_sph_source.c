FLT * X(sphzeros)(int m, int n){
	FLT * A = calloc(m * n, sizeof(FLT));
	for(int j = 0; j < n; ++j){
		for(int i = 0; i < m; ++i){
			A[i + j*m] = 0;
		}
	}
	return A;
}

void Y(print_mat)(FLT * A, char * FMT, int m, int n){
	printf("[\n");
	for (int i = 0; i < m ; ++i){
		for (int j = 0; j < n; ++j){
			printf("  ");
			printf(FMT, A[i*n + j*m]);
		}
		printf("\n");
	}
	printf("]\n");
}

// TODO: Make these functions more general as most of the code is repeated.
FLT * X(Gx)(int l){
	FLT * Gx = X(sphzeros)(2*l + 3, 2*l + 1);

	for(int k = 0; k < l-1; ++k){
		Gx[k+2 + (2*l+3)*k] = sqrt((k+1)*(k+1+1)) / (2*sqrt((2*l+1)*(2*l+3)));
		Gx[2*l-k + (2*l+3)*(2*l-k)] = Gx[k+2 + (2*l+3)*k];
	}

	for(int k = 0; k < l; ++k){
		Gx[k + (2*l+3)*k] = -1 * sqrt((2*l+2-(k+1))*(2*l+3-(k+1))) / (2 * sqrt((2*l+1)*(2*l+3)));
		Gx[2*l+2-k + (2*l+3)*(2*l-k)] = Gx[k + (2*l+3)*k] ;
	}

	Gx[(l-1)+2 + (2*l+3)*((l-1)+2)] = sqrt(2*l*(l+1)) / (2*sqrt((2*l+1)*(2*l+3)));
	Gx[(l-1)+3 + (2*l+3)*l] = -1 * sqrt(2*(l+1)*(l+2)) / (2*sqrt((2*l+1)*(2*l+3)));

	return Gx;
}

FLT * X(Gy)(int l){
	FLT * Gy = X(sphzeros)(2*l + 3, 2*l + 1);
	
	for(int k = 0; k < l-1; ++k){
		Gy[2*l-k + (2*l+3)*k] = sqrt((k+1)*(k+1+1)) / (2*sqrt((2*l+1)*(2*l+3)));
		Gy[2+k + (2*l+3)*(2*l-k)] = -1 * Gy[2*l-k + (2*l+3)*k];
	}

	for(int k = 0; k < l; ++k){
		Gy[k + (2*l+3)*(2*l-k)] = -1 * sqrt((2*l+2-(k+1))*(2*l+3-(k+1))) / (2 * sqrt((2*l+1)*(2*l+3)));
		Gy[2*l+2-k + (2*l+3)*k] = -1 * Gy[k + (2*l+3)*(2*l-k)];
	}

	Gy[(l-1)+2 + (2*l+3)*(l-1)] = sqrt(2*l*(l+1)) / (2*sqrt((2*l+1)*(2*l+3)));
	Gy[l + (2*l+3)*l] = -1 * sqrt(2*(l+1)*(l+2)) / (2*sqrt((2*l+1)*(2*l+3)));

	return Gy;
}

FLT * X(Gz)(int l){
	FLT * Gz = X(sphzeros)(2*l + 3, 2*l + 1);

	for(int k = 0; k < 2*l+1; ++k){
		Gz[k*(2*l+3) + k+1] = sqrt(((FLT)(k+1)*(2*l+2-(k+1))) / ((FLT)(2*l+1)*(2*l+3)));
	}

	return Gz;
}

// Gy has size 2l+3 by 2l+1. Then, 0 <= i <= 2l+2 and 0 <= j <= 2l.
FLT X(Gy_index)(int l, int i, int j){
	if(l+2 <= i && i <= 2*l && j == 2*l-i)
		return sqrt((FLT)(j+1)*(j+2))/(2*sqrt((FLT)(2*l+1)*(2*l+3)));
	else if(2 <= i && i <= l && j == 2*l+2-i)
		return -1 * sqrt((FLT)(i-1)*i)/(2*sqrt((FLT)(2*l+1)*(2*l+3)));
	else if(0 <= i && i <= l-1 && j == 2*l-i)
		return -1 * sqrt((FLT)(2*l+2-(i+1))*(2*l+3-(i+1)))/(2*sqrt((FLT)(2*l+1)*(2*l+3)));
	else if(l+3 <= i && i <= 2*l+2 && j == 2*l+2-i)
		return sqrt((FLT)(2*l+2-(j+1))*(2*l+3-(j+1)))/(2*sqrt((FLT)(2*l+1)*(2*l+3)));
	else if(i == l+1 && j == l-1)
		return sqrt((FLT)2*l*(l+1))/(2*sqrt((FLT)(2*l+1)*(2*l+3)));
	else if(i == l && j == l)
		return -1 * sqrt((FLT)2*(l+1)*(l+2))/(2*sqrt((FLT)(2*l+1)*(2*l+3)));

	return 0;
}

FLT * X(Gy_dense_test)(int l){
	FLT * Gyl = calloc((2*l+3)*(2*l+1), sizeof(FLT));
	
	for(int j = 0; j < 2*l+1; ++j){
		for(int i = 0; i < 2*l+3; ++i){
			Gyl[i + j*(2*l+3)] = X(Gy_index)(l, i, j);
		}
	}

	return Gyl;
}

FLT X(Y_index)(int l, int i, int j){
	FLT Yij = 0;
	for(int k = 0; k <= 2; ++k)
		Yij += X(Gy_index)(l, 2*l-i+k, i) * X(Gy_index)(l, 2*l-i+k, j);

	return Yij;
}

FLT * X(Y_dense_test)(int l){
	FLT * Yl = calloc((2*l+1)*(2*l+1), sizeof(FLT));
	for(int j = 0; j < 2*l+1; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			Yl[i + j*(2*l+1)] = X(Y_index)(l, i, j);
		}
	}

	return Yl;
}

// TODO: Check for memory leaks, specially here.
FLT * X(J)(int l){
	if(l == 0){
		FLT * A = X(sphzeros)(2*l+1, 2*l+1);
		for(int i = 0; i < 2*l+1; ++i){
			A[i + i*(2*l+1)] = 1;
		}
		return A;
	}

	FLT * Jlm1 = X(J)(l-1);
	FLT * Gylm1 = X(Gy)(l-1);
	FLT * Gzlm1 = X(Gz)(l-1);
	FLT * Gzhat = X(sphzeros)(2*(l-1)+1, 2*(l-1)+1);
	FLT * Jl = X(sphzeros)(2*l+1, 2*l+1);
	FLT * Jlv = X(sphzeros)(2*l+1, 2*l-1);

	// Initializing Gzhat.
	for(int j = 0; j < 2*(l-1)+1; ++j){
		for(int i = 0; i < 2*(l-1)+1; ++i){
			Gzhat[i + j*(2*(l-1)+1)] = Gzlm1[(i+1) + j*(2*l+1)];
		}
	}
	
	// Inverting Gzhat. Remember Gzhat is diagonal.
	for(int i = 0; i < 2*l-1; ++i)
	Gzhat[i + i*(2*l-1)] = 1/Gzhat[i + i*(2*l-1)];

	// Jlv = Gzlm1 * Jlm1
	FLT * temp = X(sphzeros)(2*l+1, 2*l-1);
	X(gemm)('N', 2*l+1, 2*l-1, 2*l-1, 1, Gylm1, 2*l+1, Jlm1, 2*l-1, 0, temp, 2*l+1);
	// Jlv = (Gzlm1 * Jlm1) * Gzhat^{-1}
	X(gemm)('N', 2*l+1, 2*l-1, 2*l-1, 1, temp, 2*l+1, Gzhat, 2*l-1, 0, Jlv, 2*l+1);

	for(int j = 1; j < 2*l; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			Jl[i + j*(2*l+1)] = Jlv[i + (j-1)*(2*l+1)];
		}
	}

	for(int i = 1; i < 2*l; ++i){
		Jl[i] = Jl[i*(2*l+1)];
		Jl[i + 2*l*(2*l+1)] = Jl[2*l + i*(2*l+1)];
	}
	Jl[2*l + 2*l*(2*l+1)] = pow(2, 1-l);

	free(Gylm1);
	free(Gzlm1);
	free(Gzhat);
	free(Jlv);
	free(temp);
	free(Jlm1);
	return Jl;
}

FLT * X(X_dense)(int l, FLT alpha){
	int n = 2*l + 1;
	FLT * Xl = X(sphzeros)(n, n);

	Xl[2*(l*l + l)] = 1;
	for(int i = 0; i < l; ++i){
		Xl[(i+l+1)*n + i+l+1] = cos((i+1) * alpha);
		Xl[(l-1-i)*n + l-1-i] = Xl[(i+l+1)*n + i+l+1];

		Xl[(l+1+i)*n + l-1-i] = sin((i+1) * alpha);
		Xl[(l-1-i)*n + l+1+i] = -1 * Xl[(l+1+i)*n + l-1-i];
	}

	return Xl;
}

// Consider upper portion of symmetric matrix.
// Lambda must be of size n. V of dimension LDA x n.
// V will return the eigenvectors.
// TODO: macros for different precisions.
void X(eigen_symmetric_lapack)(const FLT * A, int LDA, int n, FLT * Lambda, FLT * V){
	char JOBZ = 'V', UPLO = 'U';
	int LWORK = 1 + 6 * n + 2 * n * n, LIWORK = 3 + 5*n, INFO = -1;
	FLT * WORK = calloc(LWORK, sizeof(FLT));
	int * IWORK = calloc(LIWORK, sizeof(int));
	
	for(int i = 0; i < LDA; ++i){
		for(int j = i; j < n; ++j){
			V[i + j*n] = A[i + j*n];
		}
	}

	dsyevd_(&JOBZ, &UPLO, &n, V, &LDA, Lambda, WORK, &LWORK, IWORK, &LIWORK, &INFO);
	if(INFO != 0)
		printf("ERROR: problem on computing eigenvalues or eigenvectors. INFO=%d\n", INFO);
	
	free(WORK);
	free(IWORK);
}

void X(divide_Y)(FLT * Y, int l, FLT * Y1, int * n1, FLT * Y2, int * n2){
	for(int j = 0; j < *n1; ++j){
		for(int i = 0; i < *n1; ++i){
			Y1[i + j*(*n1)] = Y[2*i + 2*j*(2*l+1)];
		}
	}
	for(int j = 0; j < *n2; ++j){
		for(int i = 0; i < *n2; ++i){
			Y2[i + j*(*n2)] = Y[2*i + 1 + (2*j+1)*(2*l+1)];
		}
	}
}

// From test/test_tridiagonal.c
void Y(symmetric_tridiagonal_printmat)(char * MAT, char * FMT, ft_symmetric_tridiagonal * A) {
    int n = A->n;
    double * a = A->a;
    double * b = A->b;

    printf("%s = \n", MAT);
    if (n == 1) {
        if (a[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, a[0]);
    }
    else if (n == 2) {
        if (a[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, a[0]);
        if (b[0] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[0]);
        printf("\n");
        if (b[0] < 0) {printf(" ");}
        else {printf("  ");}
        printf(FMT, b[0]);
        if (a[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, a[1]);
    }
    else if (n > 2) {
        // First row
        if (a[0] < 0) {printf("[");}
        else {printf("[ ");}
        printf(FMT, a[0]);
        if (b[0] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[0]);
        for (int j = 2; j < n; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }

        // Second row
        printf("\n");
        if (b[0] < 0) {printf(" ");}
        else {printf("  ");}
        printf(FMT, b[0]);
        if (a[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, a[1]);
        if (b[1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[1]);
        for (int j = 3; j < n; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }

        // Interior rows
        for (int i = 2; i < n-1; i++) {
            printf("\n");
            printf("  ");
            printf(FMT, 0.0);
            for (int j = 1; j < i-1; j++) {
                printf("   ");
                printf(FMT, 0.0);
            }
            if (b[i-1] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, b[i-1]);
            if (a[i] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, a[i]);
            if (b[i] < 0) {printf("  ");}
            else {printf("   ");}
            printf(FMT, b[i]);
            for (int j = i+2; j < n; j++) {
                printf("   ");
                printf(FMT, 0.0);
            }
        }

        // Last row
        printf("\n");
        printf("  ");
        printf(FMT, 0.0);
        for (int j = 1; j < n-2; j++) {
            printf("   ");
            printf(FMT, 0.0);
        }
        if (b[n-2] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, b[n-2]);
        if (a[n-1] < 0) {printf("  ");}
        else {printf("   ");}
        printf(FMT, a[n-1]);
    }
    printf("]\n");
}

FLT * X(J_eigen)(int l){
	FLT* Gyl = X(Gy)(l);
	FLT* Y = X(sphzeros)(2*l+1, 2*l+1);

	// Y = Gyl' * Gyl
	X(gemm)('T', 2*l+3, 2*l+1, 2*l+1, 1, Gyl, 2*l+3, Gyl, 2*l+3, 0, Y, 2*l+1);
	Y(printmat)("Y", "%0.6f", Y, 2*l + 1, 2*l + 1);

	int n1 = ceil((FLT)(2*l+1)/2); 
	int n2 = 2*l+1 - n1;
	FLT * Y1 = calloc((n1)*(n1), sizeof(FLT));
	FLT * Y2 = calloc((n2)*(n2), sizeof(FLT));
	FLT * Y1_test = calloc(n1*n1, sizeof(FLT));
	FLT * Y2_test = calloc(n2*n2, sizeof(FLT));

	for(int j = 0; j < n1; ++j){
		for(int i = 0; i < n1; ++i){
			Y1_test[i + j*n1] = X(Y_index)(l, 2*i, 2*j);
		}
	}
	for(int j = 0; j < n2; ++j){
		for(int i = 0; i < n2; ++i){
			Y2_test[i + j*n2] = X(Y_index)(l, 2*i+1, 2*j+1);
		}
	}
	X(divide_Y)(Y, l, Y1, &n1, Y2, &n2);
	Y(printmat)("Y1", "%0.6f", Y1, n1, n1);
	Y(printmat)("Y2", "%0.6f", Y2, n2, n2);
	Y(printmat)("Y1_test", "%0.6f", Y1_test, n1, n1);
	Y(printmat)("Y2_test", "%0.6f", Y2_test, n2, n2);

	// TRIDIAGONAL.
	X(symmetric_tridiagonal) * Y1st = malloc(sizeof(X(symmetric_tridiagonal)));
	X(symmetric_tridiagonal) * Y2st = malloc(sizeof(X(symmetric_tridiagonal)));
	FLT * a1 = malloc(n1*sizeof(FLT));
	FLT * b1 = malloc((n1-1)*sizeof(FLT));
	FLT * a2 = malloc(n2*sizeof(FLT));
	FLT * b2 = malloc((n2-1)*sizeof(FLT));

	for(int i = 0; i < n1; ++i)
		a1[i] = X(Y_index)(l, 2*i, 2*i);
	for(int i = 0; i < n1-1; ++i)
		b1[i] = X(Y_index)(l, 2*i, 2*(i+1));
	for(int i = 0; i < n2; ++i)
		a2[i] = X(Y_index)(l, 2*i+1, 2*i+1);
	for(int i = 0; i < n2-1; ++i)
		b2[i] = X(Y_index)(l, 2*i+1, 2*(i+1)+1);
	Y1st->a = a1; Y2st->a = a2; 
	Y1st->b = b1; Y2st->b = b2;
	Y1st->n = n1; Y2st->n = n2;
	Y(symmetric_tridiagonal_printmat)("Y1st", "%0.6f", Y1st);
	Y(symmetric_tridiagonal_printmat)("Y2st", "%0.6f", Y2st);

	FLT * Y1st_lambda = calloc(n1, sizeof(FLT));
	FLT * Y2st_lambda = calloc(n2, sizeof(FLT));
	FLT * Y1st_V = X(sphzeros)(n1, n1);
	FLT * Y2st_V = X(sphzeros)(n2, n2);
	for(int i = 0 ; i < n1; ++i)
		Y1st_V[i + i*n1] = 1;
	for(int i = 0; i < n2; ++i)
		Y2st_V[i + i*n2] = 1;
	
	// Eigenvalues and eigenvectors.
	X(symmetric_tridiagonal_eig)(Y1st, Y1st_V, Y1st_lambda);
	X(symmetric_tridiagonal_eig)(Y2st, Y2st_V, Y2st_lambda);
	
	// -- ONLY FOR TESTING -- //
	printf("\n===== Test with symmetric_tridiagonal_eig =====\n");
	printf("Y1st eigenvalues = \n[");
	for(int i = 0; i < n1; ++i){
		printf("%0.6f", Y1st_lambda[i]);
		if(i < n1-1) printf("  ");
	}
	printf("]\n");
	Y(printmat)("Eigenvectors of Y1st", "%0.6f", Y1st_V, n1, n1);
	printf("Y2st eigenvalues = \n[");
	for(int i = 0; i < n2; ++i){
		printf("%0.6f", Y2st_lambda[i]);
		if(i < n2-1) printf("  ");
	}
	printf("]\n");
	Y(printmat)("Eigenvectors of Y2st", "%0.6f", Y2st_V, n2, n2);
	// ----------------- //

	// Jl from Eigenvectors of Y1 and Y2.
	FLT * Eigen_Jl = X(sphzeros)(2*l+1, 2*l+1);
	for(int j = 0; j < n2; ++j){
		for(int i = 0; i < n2; ++i){
			Eigen_Jl[2*i+1 + j*(2*l+1)] = Y2st_V[i + j*n2];
		}
	}
	for(int j = 0; j < n1; ++j){
		for(int i = 0; i < n1; ++i){
			Eigen_Jl[2*i + (j+n2)*(2*l+1)] = Y1st_V[i + (n1-1-j)*n1];
		}
	}

	// -- ONLY FOR TESTING -- //
	Y(printmat)("Eigen_Jl", "%0.6f", Eigen_Jl, 2*l+1, 2*l+1);

	// Constructing Zl from Yl eigenvalues.
	FLT * Zl = X(sphzeros)(2*l+1, 2*l+1);
	for(int i = 0; i < n2; ++i)
		Zl[i + i*(2*l+1)] = Y2st_lambda[i];
	for(int i = 0; i < n1; ++i)
		Zl[(i+n2) + (i+n2)*(2*l+1)] = Y1st_lambda[n1-1-i];
	//Y(printmat)("Zl", "%0.6f", Zl, 2*l+1, 2*l+1);
	// ----------------- //


	free(Gyl);
	free(Y1);
	free(Y2);
	return Eigen_Jl;
}

void X(do_a_test()){
	int l = 3;
	
	FLT * Jl_eigen = X(J_eigen)(l);

	FLT * Slow_Jl = X(J)(l);
	Y(printmat)("Slow_Jl", "%0.6f", Slow_Jl, 2*l+1, 2*l+1);

	FLT * Gy = X(Gy)(l);
	FLT * Gy_test = X(Gy_dense_test)(l);
	//Y(printmat)("Gy", "%0.6f", Gy, 2*l+3, 2*l+1);
	//Y(printmat)("Gy_test", "%0.6f", Gy_test, 2*l+3, 2*l+1);

	FLT * Yl_test = X(Y_dense_test)(l);
	//Y(printmat)("Y_test", "%0.6f", Yl_test, 2*l+1, 2*l+1);

	free(Jl_eigen);
	free(Slow_Jl);

	// Testing FMM.
	// Gy is anti-banded.
	//FLT* Gyl = X(Gy)(l);
	//FLT* Y = X(sphzeros)(2*l+1, 2*l+1);

	//// Y = Gyl' * Gyl
	//X(gemm)('T', 2*l+3, 2*l+1, 2*l+1, 1, Gyl, 2*l+3, Gyl, 2*l+3, 0, Y, 2*l+1);

	//int n1 = ceil((FLT)(2*l+1)/2); 
	//int n2 = 2*l+1 - n1;
	//FLT * Y1 = calloc((n1)*(n1), sizeof(FLT));
	//FLT * Y2 = calloc((n2)*(n2), sizeof(FLT));
	//X(divide_Y)(Y, l, Y1, &n1, Y2, &n2);

	//X(symmetric_tridiagonal) stY1 = malloc(sizeof(X(symmetric_tridiagonal)));
	//X(symmetric_tridiagonal) stY2 = malloc(sizeof(X(symmetric_tridiagonal)));
	//FLT * a1 = malloc(n1*sizeof(FLT));
	//FLT * a2 = malloc(n2*sizeof(FLT));
	//FLT * b1 = malloc((n1-1)*sizeof(FLT));
	//FLT * b2 = malloc((n2-1)*sizeof(FLT));

	
}


