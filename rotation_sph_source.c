FLT * X(sphzeros)(int m, int n){
	FLT * A = (FLT*)calloc(m * n,  sizeof(FLT));
	for(int j = 0; j < n; ++j){
		for(int i = 0; i < m; ++i){
			A[i + j*m] = 0;
		}
	}
	return A;
}

void X(threshold)(FLT * A, int m, int n, FLT eps){
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < n; ++j){
			if(Y(fabs)(A[i + j*m]) < eps)
				A[i + j*m] = 0;
		}
	}
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

FLT X(Gz_index)(int l, int i, int j){
	if(i == j+1)
		return sqrt((j+1)*(2*l+2-(j+1)))/sqrt((2*l+1)*(2*l+3));
	return 0;
}

FLT X(Gzhat_index)(int l, int i, int j){
	return X(Gz_index)(l, i+1, j);
}

FLT X(Gzhatinv_index)(int l, int i, int j){
	if(i == j)
		return 1/X(Gzhat_index)(l, i, j);
	return 0;
}

FLT * X(Gz_dense_test)(int l, int option){
	FLT * Gz = NULL;
	int rows = 0;
	if(option == 0){
		Gz = X(sphzeros)(2*l+3, 2*l+1);
		rows = 2*l+3;
	}
	else{
		Gz = X(sphzeros)(2*l+1, 2*l+1);
		rows = 2*l+1;
	}

	for(int j = 0; j < 2*l+1; ++j){
		for(int i = 0; i < rows; ++i){
			if(option == 0)
				Gz[i + j*rows] = X(Gz_index)(l, i, j);
			if(option == 1)
				Gz[i + j*rows] = X(Gzhat_index)(l, i, j);
			if(option == 2)
				Gz[i + j*rows] = X(Gzhatinv_index)(l, i, j);
		}
	}

	return Gz;
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
	FLT * WORK = (FLT*)calloc(LWORK, sizeof(FLT));
	int * IWORK = (int*)calloc(LIWORK, sizeof(int));
	
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

// Correct the sign of jth column of J using the sign pattern.
// Odd and even here refer to indices starting at zero.
// In case symmetry != 0, only strict lower triangular part will be corrected.
void X(correct_J_column_sign)(FLT * J, int l, int j, int symmetry){
	int begin = -1, end = -1, step = 0;
	if(j < l){ // First half.
		if(j % 2 == 0){ // Zero from row 0 to l-1.
			begin = 2*l-1;
			end = (symmetry != 0 ? j : l);
			step = -2;
		}
		else{ // Zero from row l to 2l.
			begin = (symmetry != 0 ? j+2 : 1);
			end = l;
			step = 2;
		}
	}
	else{ // j >= l.
		if(j % 2 == 0){ // Even rows.
			begin = 2*l;
			end = (symmetry != 0 ? j : l);
			step = -2;
		}
		else{ 
			begin = (symmetry != 0 ? j+2 : 0);
			end = l;
			step = 2;
		}
	}

	int k = begin;
	while(1){
		if(step == -2 && k < end){
			break;
		}
		else if(step == 2 && k > end){
			break;
		}

		if(symmetry != 0){
			if((J[k + j*(2*l+1)] < 0 ? 1 : 0) != (J[j + k*(2*l+1)] < 0))
				J[k + j*(2*l+1)] = J[j + k*(2*l+1)];
		}
		else{
			J[k + j*(2*l+1)] = -1 * J[k + j*(2*l+1)];
		}
		k = k + step;
	}
}

void X(correct_J_column_signs)(FLT * J, int l, int j){
	X(correct_J_column_sign)(J, l, j, 0);
}

void X(correct_J_column_symmetry)(FLT * J, int l, int j){
	X(correct_J_column_sign)(J, l, j, 1);
}

FLT * X(J_eigen)(int l){
	int n1 = ceil((FLT)(2*l+1)/2); 
	int n2 = 2*l+1 - n1;
	X(symmetric_tridiagonal) * Y1st = (X(symmetric_tridiagonal)*)malloc(sizeof(X(symmetric_tridiagonal)));
	X(symmetric_tridiagonal) * Y2st = (X(symmetric_tridiagonal)*)malloc(sizeof(X(symmetric_tridiagonal)));
	FLT * a1 = (FLT*)calloc(n1, sizeof(FLT));
	FLT * b1 = (FLT*)calloc((n1-1), sizeof(FLT));
	FLT * a2 = (FLT*)calloc(n2, sizeof(FLT));
	FLT * b2 = (FLT*)calloc((n2-1), sizeof(FLT));

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

	FLT * Y1st_lambda = (FLT*)calloc(n1, sizeof(FLT));
	FLT * Y2st_lambda = (FLT*)calloc(n2, sizeof(FLT));
	FLT * Y1st_V = X(sphzeros)(n1, n1);
	FLT * Y2st_V = X(sphzeros)(n2, n2);
	for(int i = 0 ; i < n1; ++i)
		Y1st_V[i + i*n1] = 1;
	for(int i = 0; i < n2; ++i)
		Y2st_V[i + i*n2] = 1;
	
	// Eigenvalues and eigenvectors.
	X(symmetric_tridiagonal_eig)(Y1st, Y1st_V, Y1st_lambda);
	X(symmetric_tridiagonal_eig)(Y2st, Y2st_V, Y2st_lambda);

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

	// Correcting symmetry.
	//for(int j = 0; j < 2*l-1; ++j){
	//	X(correct_J_column_symmetry)(Eigen_Jl, l, j);
	//}
	
	// Correcting signs. Evaluating the sign pattern.
	int l_parity = l%2;
	for(int j = l + l_parity; j < 2*l+1; j = j+2){ // Bottom right block.
		if((Eigen_Jl[l + l_parity + j*(2*l+1)] < 0 ? 1 : 0) != (pow(-1, floor((FLT)l/2)) < 0 ? 1 : 0))
			X(correct_J_column_signs)(Eigen_Jl, l, j);
	}

	for(int j = l + (1-l_parity); j < 2*l+1; j = j+2){ // Top right block.
		if((Eigen_Jl[l-1-(1-l_parity) + j*(2*l+1)] < 0 ? 1 : 0) != (pow(-1, ceil((FLT)l/2)) < 0 ? 1 : 0))
			X(correct_J_column_signs)(Eigen_Jl, l, j);
	}

	for(int j = 0; j < l; j = j+2){ // Bottom left block.
		if((Eigen_Jl[l+(1-l_parity) + j*(2*l+1)] < 0 ? 1 : 0) != (pow(-1, ceil((FLT)l/2)) < 0 ? 1 : 0))
			X(correct_J_column_signs)(Eigen_Jl, l, j);
	}

	for(int j = 1; j < l; j = j+2){ // Top left block.
		if((Eigen_Jl[l-1-l_parity + j*(2*l+1)] < 0 ? 1 : 0) != (pow(-1, floor((FLT)l/2)+1) < 0 ? 1 : 0))
			X(correct_J_column_signs)(Eigen_Jl, l, j);
	}

	X(threshold)(Eigen_Jl, 2*l+1, 2*l+1, Y(eps)());

	free(Y1st_lambda);
	free(Y2st_lambda);
	free(Y1st_V);
	free(Y2st_V);
	X(destroy_symmetric_tridiagonal)(Y1st);
	X(destroy_symmetric_tridiagonal)(Y2st);
	return Eigen_Jl;
}

// FIXME: __cospi function is different for long double and quadruple.
FLT X(X_index)(int l, FLT alpha, int i, int j){
	if(i == j && i == l)
		return 1;
	else if(i == j)
		return __cospi((l-i)*alpha);
	else if(j == 2*l-i)
		return __sinpi((l-i)*alpha);

	return 0;
}

FLT * X(X_test)(int l, FLT alpha){
	FLT * Xl = X(sphzeros)(2*l+1, 2*l+1);

	for(int j = 0; j < 2*l+1; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			Xl[i + j*(2*l+1)] = X(X_index)(l, alpha, i, j);
		}
	}
	return Xl;
}

FLT X(JXJX)(FLT * J, int l, int i, int j, FLT beta, FLT gamma){
	FLT entry = 0;
	for(int k = 0; k < 2*l+1; ++k){
		if(j == l && k == l)
			entry += J[i + k*(2*l+1)] * J[k + j*(2*l+1)];
		else if(j != l && k == l)
			entry += J[i + k*(2*l+1)] * (J[k + j*(2*l+1)]*X(X_index)(l, gamma, j, j) + J[k + (2*l-j)*(2*l+1)]*X(X_index)(l, gamma, 2*l-j, j));
		else if(j == l && k != l)
			entry += (J[i + k*(2*l+1)]*X(X_index)(l, beta, k, k) + J[i + (2*l-k)*(2*l+1)]*X(X_index)(l, beta, 2*l-k, k)) * J[k + j*(2*l+1)];
		else 
			entry += (J[i + k*(2*l+1)]*X(X_index)(l, beta, k, k) + J[i + (2*l-k)*(2*l+1)]*X(X_index)(l, beta, 2*l-k, k)) * 
					 (J[k + j*(2*l+1)]*X(X_index)(l, gamma, j, j) + J[k + (2*l-j)*(2*l+1)]*X(X_index)(l, gamma, 2*l-j, j));
	}	 

	return entry;
}

FLT * X(rotation_matrix_J)(int l, FLT alpha, FLT beta, FLT gamma, FLT * J){
	FLT * Delta = X(sphzeros)(2*l+1, 2*l+1);

	for(int j = 0; j < 2*l+1; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			FLT entry = 0;

			entry = X(X_index)(l, alpha, i, i)*X(JXJX)(J, l, i, j, beta, gamma);
			if(i != l)
				entry += X(X_index)(l, alpha, i, 2*l-i)*X(JXJX)(J, l, 2*l-i, j, beta, gamma);

			Delta[i + j*(2*l+1)] = entry;
		}
	}

	return Delta;
}

FLT * X(rotation_matrix_direct)(int l, FLT alpha, FLT beta, FLT gamma){
	FLT * J = X(J)(l);
	return X(rotation_matrix_J)(l, alpha, beta, gamma, J);
}

FLT * X(rotation_matrix)(int l, FLT alpha, FLT beta, FLT gamma){
	FLT * J = X(J_eigen)(l);
	return X(rotation_matrix_J)(l, alpha, beta, gamma, J);
}

void X(do_a_test()){
	int l = 5;
	
	FLT * Eigen_Jl = X(J_eigen)(l);
	FLT * Slow_Jl = X(J)(l);

	//Y(printmat)("Slow_Jl", "%0.6f", Slow_Jl, 2*l+1, 2*l+1);
	//Y(printmat)("Eigen_Jl", "%0.6f", Eigen_Jl, 2*l+1, 2*l+1);
	printf("Test eigen = %0.30f\n", Eigen_Jl[167 + 57*(2*l+1)]);
	printf("Test slow = %0.30f\n", Slow_Jl[167 + 57*(2*l+1)]);

	// Test Gz_index.
	FLT * Gz = X(Gz_dense_test)(l, 0);
	FLT * Gzhat = X(Gz_dense_test)(l, 1);
	FLT * Gzhatinv = X(Gz_dense_test)(l, 2);
	Y(printmat)("Gz", "%0.6f", Gz, 2*l+3, 2*l+1);
	Y(printmat)("Gzhat", "%0.6f", Gzhat, 2*l+1, 2*l+1);
	Y(printmat)("Gzhatinv", "%0.6f", Gzhatinv, 2*l+1, 2*l+1);

	FLT alpha = 0.123;
	FLT beta = 0.456;
	FLT gamma = 0.789;
	FLT * delta = X(rotation_matrix_direct)(l, alpha, beta, gamma);
	//Y(printmat)("Delta", "%0.6f", delta, 2*l+1, 2*l+1);

	free(Eigen_Jl);
	free(Slow_Jl);

	//int number_of_tests = 170;
	//for(int k = 170; k <= number_of_tests; ++k){
	//	FLT * _J_eigen = X(J_eigen)(k);
	//	FLT * _J = X(J)(k);

	//	printf("\rTest number k = %d", k);
	//	for(int j = 0; j < 2*k+1; ++j){
	//		for(int i = 0; i < 2*k+1; ++i){
	//			if(abs(_J_eigen[i + j*(2*k+1)] - _J[i + j*(2*k+1)]) > 1e-14){
	//				printf("\nERROR! Not identical entries for [%d, %d].\n", i, j);
	//				printf("Eigen_Jl[i, j] = %.30f\n", _J_eigen[i + j*(2*k+1)]);
	//				printf("J[i, j] = %.30f\n", _J[i + j*(2*k+1)]);
	//				//return;
	//			}
	//		}
	//	}
	//	free(_J_eigen);
	//	free(_J);
	//}
	//printf("\nTest done!\n");

	//int number_of_tests = 10000;
	//for(int k = 1; k <= number_of_tests; ++k){
	//	FLT * _Eigen_Jl = X(J_eigen)(k);
	//
	//	// Testing symmetry. (Slow)
	//	for(int j = 0; j < 2*k+1; ++j){
	//		for(int i = 0; i < 2*k+1; ++i){
	//			if(_Eigen_Jl[i + j*(2*k+1)] - _Eigen_Jl[j + i*(2*k+1)] > 1e-8){
	//				printf("ERROR: Eigen_Jl not symmetric for k = %d and [%d, %d]\n", k, i, j);
	//				printf("Eigen_Jl[i, j] = %.30f\n", _Eigen_Jl[i + j*(2*k+1)]);
	//				printf("Eigen_Jl[j, i] = %.30f\n\n", _Eigen_Jl[j + i*(2*k+1)]);
	//				return;
	//			}
	//		}
	//	}
	//	
	//	free(_Eigen_Jl);
	//}
	//printf("Test done.\n");
}
