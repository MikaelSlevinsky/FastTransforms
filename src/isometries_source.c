FLT * X(Gx)(int l){
	FLT * Gx = (FLT*)calloc((2*l + 3)*(2*l + 1), sizeof(FLT));

	for(int k = 0; k < l-1; ++k){
		Gx[k+2 + (2*l+3)*k] = Y(sqrt)((k+1)*(k+1+1)) / (2*Y(sqrt)((2*l+1)*(2*l+3)));
		Gx[2*l-k + (2*l+3)*(2*l-k)] = Gx[k+2 + (2*l+3)*k];
	}

	for(int k = 0; k < l; ++k){
		Gx[k + (2*l+3)*k] = -1 * Y(sqrt)((2*l+2-(k+1))*(2*l+3-(k+1))) / (2 * Y(sqrt)((2*l+1)*(2*l+3)));
		Gx[2*l+2-k + (2*l+3)*(2*l-k)] = Gx[k + (2*l+3)*k] ;
	}

	Gx[(l-1)+2 + (2*l+3)*((l-1)+2)] = Y(sqrt)(2*l*(l+1)) / (2*Y(sqrt)((2*l+1)*(2*l+3)));
	Gx[(l-1)+3 + (2*l+3)*l] = -1 * Y(sqrt)(2*(l+1)*(l+2)) / (2*Y(sqrt)((2*l+1)*(2*l+3)));

	return Gx;
}

FLT * X(Gy)(int l){
	FLT * Gy = (FLT*)calloc((2*l + 3)*(2*l + 1), sizeof(FLT));
	
	for(int k = 0; k < l-1; ++k){
		Gy[2*l-k + (2*l+3)*k] = Y(sqrt)((k+1)*(k+1+1)) / (2*Y(sqrt)((2*l+1)*(2*l+3)));
		Gy[2+k + (2*l+3)*(2*l-k)] = -1 * Gy[2*l-k + (2*l+3)*k];
	}

	for(int k = 0; k < l; ++k){
		Gy[k + (2*l+3)*(2*l-k)] = -1 * Y(sqrt)((2*l+2-(k+1))*(2*l+3-(k+1))) / (2 * Y(sqrt)((2*l+1)*(2*l+3)));
		Gy[2*l+2-k + (2*l+3)*k] = -1 * Gy[k + (2*l+3)*(2*l-k)];
	}

	Gy[(l-1)+2 + (2*l+3)*(l-1)] = Y(sqrt)(2*l*(l+1)) / (2*Y(sqrt)((2*l+1)*(2*l+3)));
	Gy[l + (2*l+3)*l] = -1 * Y(sqrt)(2*(l+1)*(l+2)) / (2*Y(sqrt)((2*l+1)*(2*l+3)));

	return Gy;
}

FLT * X(Gz)(int l){
	FLT * Gz = (FLT*)calloc((2*l + 3)*(2*l + 1), sizeof(FLT));

	for(int k = 0; k < 2*l+1; ++k){
		Gz[k*(2*l+3) + k+1] = Y(sqrt)(((FLT)(k+1)*(2*l+2-(k+1))) / ((FLT)(2*l+1)*(2*l+3)));
	}

	return Gz;
}

// Gy has size 2l+3 by 2l+1. Then, 0 <= i <= 2l+2 and 0 <= j <= 2l.
FLT X(Gy_index)(int l, int i, int j){
	if(l+2 <= i && i <= 2*l && j == 2*l-i)
		return Y(sqrt)((FLT)(j+1)*(j+2))/(2*Y(sqrt)((FLT)(2*l+1)*(2*l+3)));
	else if(2 <= i && i <= l && j == 2*l+2-i)
		return -1 * Y(sqrt)((FLT)(i-1)*i)/(2*Y(sqrt)((FLT)(2*l+1)*(2*l+3)));
	else if(0 <= i && i <= l-1 && j == 2*l-i)
		return -1 * Y(sqrt)((FLT)(2*l+2-(i+1))*(2*l+3-(i+1)))/(2*Y(sqrt)((FLT)(2*l+1)*(2*l+3)));
	else if(l+3 <= i && i <= 2*l+2 && j == 2*l+2-i)
		return Y(sqrt)((FLT)(2*l+2-(j+1))*(2*l+3-(j+1)))/(2*Y(sqrt)((FLT)(2*l+1)*(2*l+3)));
	else if(i == l+1 && j == l-1)
		return Y(sqrt)((FLT)2*l*(l+1))/(2*Y(sqrt)((FLT)(2*l+1)*(2*l+3)));
	else if(i == l && j == l)
		return -1 * Y(sqrt)((FLT)2*(l+1)*(l+2))/(2*Y(sqrt)((FLT)(2*l+1)*(2*l+3)));

	return 0;
}

FLT X(Gz_index)(int l, int i, int j){
	if(i == j+1)
		return Y(sqrt)((j+1)*(2*l+2-(j+1)))/Y(sqrt)((2*l+1)*(2*l+3));
	return 0;
}

FLT X(Gzhat_index)(int l, int i, int j){
	return X(Gz_index)(l, i+1, j);
}

FLT X(Gzhatinv_index)(int l, int i, int j){
	if(i == j)
		return ONE(FLT)/X(Gzhat_index)(l, i, j);
	return 0;
}

FLT X(X_index)(int l, FLT alpha, int i, int j){
	if(i == j && i == l)
		return 1;
	else if(i == j)
		return Y(__cospi)((l-i)*alpha);
	else if(j == 2*l-i)
		return Y(__sinpi)((l-i)*alpha);

	return 0;
}

FLT X(Y_index)(int l, int i, int j){
	FLT Yij = ZERO(FLT);
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

FLT X(Z_index)(int l, int i, int j){
	if(i != j)
		return ZERO(FLT);

	return Y(pow)(X(Gzhat_index)(l, i, i), 2);
}

FLT * X(Z_diagonal)(int l){
	FLT * diagonal = calloc(2*l+1, sizeof(FLT));
	for(int i = 0; i < 2*l+1; ++i)
		diagonal[i] = X(Z_index)(l, i, i);

	return diagonal;
}

int X(is_J_entry_nonzero)(int l, int i, int j){
	if(j < l){ // Left blocks.
		if(j%2 == 0){
			if(i >= l && i%2 == 1)
				return 1;
		}
		else{
			if(i < l && i%2 == 1)
				return 1;
		}
	}
	else{ // Right blocks.
		if(j%2 == 0){
			if(i >= l && i%2 == 0)
				return 1;
		}
		else{
			if(i < l && i%2 == 0)
				return 1;
		}
	}

	return 0;
}

FLT * X(J)(int l){
	if(l == 0){
		FLT * A = (FLT*)calloc((2*l+1)*(2*l+1), sizeof(FLT));
		for(int i = 0; i < 2*l+1; ++i){
			A[i + i*(2*l+1)] = ONE(FLT);
		}
		return A;
	}

	FLT * Jlm1 = X(J)(l-1);
	FLT * Jl = (FLT*)calloc((2*l+1)*(2*l+1), sizeof(FLT));
	FLT * Jlv = (FLT*)calloc((2*l+1)*(2*l-1), sizeof(FLT));

	// Jlv = Gylm1 * Jlm1 * Gzhatinv_lm1
	for(int j = 0; j < 2*l-1; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			FLT entry = ZERO(FLT);

			if(X(is_J_entry_nonzero)(l-1, 2*(l-1)-i, j) == 1 && i < 2*l-1)
				entry += X(Gy_index)(l-1, i, 2*(l-1)-i) * Jlm1[2*(l-1)-i + j*(2*l-1)] * X(Gzhatinv_index)(l-1, j, j);
			if(X(is_J_entry_nonzero)(l-1, 2*l-i, j) == 1 && i > 1)
				entry += X(Gy_index)(l-1, i, 2*l-i) * Jlm1[2*l-i + j*(2*l-1)] * X(Gzhatinv_index)(l-1, j, j);

			Jlv[i + j*(2*l+1)] = entry;
		}
	}

	for(int j = 1; j < 2*l; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			Jl[i + j*(2*l+1)] = Jlv[i + (j-1)*(2*l+1)];
		}
	}

	for(int i = 1; i < 2*l; ++i){
		Jl[i] = Jl[i*(2*l+1)];
		Jl[i + 2*l*(2*l+1)] = Jl[2*l + i*(2*l+1)];
	}
	Jl[2*l + 2*l*(2*l+1)] = Y(pow)(2, 1-l);

	free(Jlv);
	free(Jlm1);
	return Jl;
}

FLT * X(J_eigen)(int l){
	int n1 = floor((FLT)(l+1)/2); int n2 = l+1 - n1;
	int n3 = floor((FLT)l/2); int n4 = l - n3;
	X(symmetric_tridiagonal) * Y1 = (X(symmetric_tridiagonal)*)malloc(sizeof(X(symmetric_tridiagonal)));
	X(symmetric_tridiagonal) * Y2 = (X(symmetric_tridiagonal)*)malloc(sizeof(X(symmetric_tridiagonal)));
	X(symmetric_tridiagonal) * Y3 = (X(symmetric_tridiagonal)*)malloc(sizeof(X(symmetric_tridiagonal)));
	X(symmetric_tridiagonal) * Y4 = (X(symmetric_tridiagonal)*)malloc(sizeof(X(symmetric_tridiagonal)));
	FLT * a1 = (FLT*)malloc(n1 * sizeof(FLT)); FLT * b1 = (FLT*)malloc((n1-1) * sizeof(FLT));
	FLT * a2 = (FLT*)malloc(n2 * sizeof(FLT)); FLT * b2 = (FLT*)malloc((n2-1) * sizeof(FLT));
	FLT * a3 = (FLT*)malloc(n3 * sizeof(FLT)); FLT * b3 = (FLT*)malloc((n3-1) * sizeof(FLT));
	FLT * a4 = (FLT*)malloc(n4 * sizeof(FLT)); FLT * b4 = (FLT*)malloc((n4-1) * sizeof(FLT));

	// Populating vectors.
	for(int i = 0; i < n1; ++i)
		a1[i] = X(Y_index)(l, 2*i, 2*i);
	for(int i = 0; i < n1-1; ++i)
		b1[i] = X(Y_index)(l, 2*(i+1), 2*i);
	for(int i = n1; i < n1+n2; ++i)
		a2[n2-1-(i-n1)] = X(Y_index)(l, 2*i, 2*i);
	for(int i = n1; i < n1+n2-1; ++i)
		b2[n2-2-(i-n1)] = X(Y_index)(l, 2*(i+1), 2*i);
	for(int i = 0; i < n3; ++i)
		a3[i] = X(Y_index)(l, 2*i+1, 2*i+1);
	for(int i = 0; i < n3-1; ++i)
		b3[i] = X(Y_index)(l, 2*(i+1)+1, 2*i+1);
	for(int i = n3; i < n3+n4; ++i)
		a4[n4-1-(i-n3)] = X(Y_index)(l, 2*i+1, 2*i+1);
	for(int i = n3; i < n3+n4-1; ++i)
		b4[n4-2-(i-n3)] = X(Y_index)(l, 2*(i+1)+1, 2*i+1);

	Y1->a = a1; Y3->a = a3;
	Y1->b = b1; Y3->b = b3;
	Y1->n = n1; Y3->n = n3;
	Y2->a = a2; Y4->a = a4;
	Y2->b = b2; Y4->b = b4;
	Y2->n = n2; Y4->n = n4;

	// Known eigenvalues.
	FLT * lambda1 = (FLT*)calloc(n1, sizeof(FLT));
	FLT * lambda2 = (FLT*)calloc(n2, sizeof(FLT));
	FLT * lambda3 = (FLT*)calloc(n3, sizeof(FLT));
	FLT * lambda4 = (FLT*)calloc(n4, sizeof(FLT));

	for(int i = 0; i < n1; ++i)
		lambda1[i] = X(Z_index)(l, 2*i+1, 2*i+1);
	for(int i = 0; i < n2; ++i)
		lambda2[i] = X(Z_index)(l, 2*i, 2*i);
	for(int i = 0; i < n3; ++i)
		lambda3[i] = X(Z_index)(l, 2*l-1 - 2*i, 2*l-1 - 2*i);
	for(int i = 0; i < n4; ++i)
		lambda4[i] = X(Z_index)(l, 2*l - 2*i, 2*l - 2*i);

	X(symmetric_tridiagonal_symmetric_eigen) * Y1stse = X(symmetric_tridiagonal_symmetric_eig)(Y1, lambda1, 1);
	X(symmetric_tridiagonal_symmetric_eigen) * Y2stse = X(symmetric_tridiagonal_symmetric_eig)(Y2, lambda2, 1);
	X(symmetric_tridiagonal_symmetric_eigen) * Y3stse = X(symmetric_tridiagonal_symmetric_eig)(Y3, lambda3, 1);
	X(symmetric_tridiagonal_symmetric_eigen) * Y4stse = X(symmetric_tridiagonal_symmetric_eig)(Y4, lambda4, 1);

	FLT * temp = Y1stse->phi0;
	Y1stse->phi0 = Y4stse->phi0;
	Y4stse->phi0 = temp;

	X(densematrix) * V1 = (X(densematrix)*)malloc(sizeof(X(densematrix)));
	X(densematrix) * V2 = (X(densematrix)*)malloc(sizeof(X(densematrix)));
	X(densematrix) * V3 = (X(densematrix)*)malloc(sizeof(X(densematrix)));
	X(densematrix) * V4 = (X(densematrix)*)malloc(sizeof(X(densematrix)));
	V1->m = n1; V3->m = n3;
	V1->n = n1; V3->n = n3;
	V2->m = n2; V4->m = n4;
	V2->n = n2; V4->n = n4;

	V1->A = X(symmetric_tridiagonal_symmetric_eigenvectors)(Y1stse);
	V2->A = X(symmetric_tridiagonal_symmetric_eigenvectors)(Y2stse);
	V3->A = X(symmetric_tridiagonal_symmetric_eigenvectors)(Y3stse);
	V4->A = X(symmetric_tridiagonal_symmetric_eigenvectors)(Y4stse);

	// Sign correction.
	FLT * x = (FLT*)calloc(n1, sizeof(FLT));
	for(int i = 0; i < n1; ++i)
		x[i] = ONE(FLT);
	for(int j = 0; j < n1; ++j)
		if((V1->A[(n1-1) + j*n1] < 0 ? 1 : 0) != (pow(-1, ceil((FLT)l/2)) < 0 ? 1 : 0))
			X(scale_columns_densematrix)(-1, x, V1);
	for(int j = 0; j < n2; ++j)
		if((V2->A[n2-1 + j*n2] < 0 ? 1 : 0) != (pow(-1, floor((FLT)l/2)) < 0 ? 1 : 0))
			X(scale_columns_densematrix)(-1, x, V2);
	for(int j = 0; j < n3; ++j)
		if((V3->A[(n3-1) + j*n3] < 0 ? 1 : 0) != (pow(-1, floor((FLT)l/2)+1) < 0 ? 1 : 0))
			X(scale_columns_densematrix)(-1, x, V3);
	for(int j = 0; j < n4; ++j)
		if((V4->A[n4-1 + j*n4] < 0 ? 1 : 0) != (pow(-1, ceil((FLT)l/2)) < 0 ? 1 : 0))
			X(scale_columns_densematrix)(-1, x, V4);

	// Building Jl.
	FLT * Jl = (FLT*)calloc((2*l+1)*(2*l+1), sizeof(FLT));
	for(int j = 0; j < n1; ++j)
		for(int i = 0; i < n1; ++i)
			Jl[2*l-1-2*i + 2*j*(2*l+1)] = V1->A[i + j*n1];
	for(int j = 0; j < n2; ++j)
		for(int i = 0; i < n2; ++i)
			Jl[2*l-2*i + (2*l-2*j)*(2*l+1)] = V2->A[i + j*n2];
	for(int j = 0; j < n3; ++j)
		for(int i = 0; i < n3; ++i)
			Jl[2*i+1 + (2*j+1)*(2*l+1)] = V3->A[i + j*n3];
	for(int j = 0; j < n4; ++j)
		for(int i = 0; i < n4; ++i)
			Jl[2*i + (2*l-1-2*j)*(2*l+1)] = V4->A[i + j*n4];

	free(lambda1); free(lambda3);
	free(lambda2); free(lambda4);
	X(destroy_densematrix)(V1);
	X(destroy_densematrix)(V2);
	X(destroy_densematrix)(V3);
	X(destroy_densematrix)(V4);
	X(destroy_symmetric_tridiagonal)(Y1);
	X(destroy_symmetric_tridiagonal)(Y2);
	X(destroy_symmetric_tridiagonal)(Y3);
	X(destroy_symmetric_tridiagonal)(Y4);
	X(destroy_symmetric_tridiagonal_symmetric_eigen)(Y1stse);
	X(destroy_symmetric_tridiagonal_symmetric_eigen)(Y2stse);
	X(destroy_symmetric_tridiagonal_symmetric_eigen)(Y3stse);
	X(destroy_symmetric_tridiagonal_symmetric_eigen)(Y4stse);
	return Jl;
}

FLT X(JXJX)(FLT * J, int l, int i, int j, FLT beta, FLT gamma){
	FLT entry = ZERO(FLT);
	for(int k = 0; k < 2*l+1; ++k){
		if(X(is_J_entry_nonzero)(l, i, k) == 1 && X(is_J_entry_nonzero)(l, k, j) == 1 && j == l && k == l)
			entry += J[i + k*(2*l+1)] * J[k + j*(2*l+1)];
		else if(X(is_J_entry_nonzero)(l, i, k) == 1 && j != l && k == l)
			entry += J[i + k*(2*l+1)] * (J[k + j*(2*l+1)]*X(X_index)(l, gamma, j, j) + J[k + (2*l-j)*(2*l+1)]*X(X_index)(l, gamma, 2*l-j, j));
		else if(X(is_J_entry_nonzero)(l, k, j) == 1 && j == l && k != l)
			entry += (J[i + k*(2*l+1)]*X(X_index)(l, beta, k, k) + J[i + (2*l-k)*(2*l+1)]*X(X_index)(l, beta, 2*l-k, k)) * J[k + j*(2*l+1)];
		else 
			entry += (J[i + k*(2*l+1)]*X(X_index)(l, beta, k, k) + J[i + (2*l-k)*(2*l+1)]*X(X_index)(l, beta, 2*l-k, k)) * 
					 (J[k + j*(2*l+1)]*X(X_index)(l, gamma, j, j) + J[k + (2*l-j)*(2*l+1)]*X(X_index)(l, gamma, 2*l-j, j));
	}	 

	return entry;
}

FLT * X(rotation_matrix_J)(int l, FLT alpha, FLT beta, FLT gamma, FLT * J){
	FLT * Delta = (FLT*)calloc((2*l+1)*(2*l+1), sizeof(FLT));
	// Fasttransforms uses Condon-Shortley phase convention.
	beta *= -ONE(FLT);

	for(int j = 0; j < 2*l+1; ++j){
		for(int i = 0; i < 2*l+1; ++i){
			FLT entry = ZERO(FLT);

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

