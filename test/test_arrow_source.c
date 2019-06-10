FLT X(secular_cond)(X(symmetric_arrow) * A, FLT lambda) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, ret = X(fabs)(lambda)+X(fabs)(A->c), t;
    for (int i = 0; i < n-1; i++) {
        t = b[i];
        t = t*t/(a[i]-lambda);
        ret += X(fabs)(t);
    }
    return ret;
}

FLT X(secular_derivative_cond)(X(symmetric_arrow) * A, FLT lambda) {return X(secular_derivative)(A, lambda);}

FLT X(secular_second_derivative_cond)(X(symmetric_arrow) * A, FLT lambda) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, ret = ZERO(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = b[i]/(a[i]-lambda);
        t = t*t/(a[i]-lambda);
        ret += X(fabs)(t);
    }
    return TWO(FLT)*ret;
}


void X(test_arrow)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    int n = 50;
    struct timeval start, end;

    X(symmetric_arrow) * A = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    A->a = (FLT *) calloc(n-1, sizeof(FLT));
    A->b = (FLT *) calloc(n-1, sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        A->a[i] = (i+1)*(i+1);
        A->b[i] = i+1;
    }
    A->c = (n*n+n-1);
    A->n = n;
    X(upper_arrow) * R = X(symmetric_arrow_cholesky)(A);
    FLT * d_true = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * e_true = (FLT *) calloc(n-1, sizeof(FLT));
    FLT f_true = n;
    for (int i = 0; i < n-1; i++) {
        d_true[i] = i+1;
        e_true[i] = 1;
    }
    FLT err = X(sqrt)(X(pow)(X(norm_2arg)(R->d, d_true, n-1), 2) + X(pow)(X(norm_2arg)(R->e, e_true, n-1), 2) + X(pow)(R->f-f_true, 2))/X(sqrt)(X(pow)(X(norm_1arg)(d_true, n-1), 2) + X(pow)(X(norm_1arg)(e_true, n-1), 2) + X(pow)(f_true, 2));
    printf("Frobenius error in the Cholesky factor \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    FLT alpha = 2;
    FLT beta = 0.5;
    FLT * x = (FLT *) calloc(n, sizeof(FLT));
    FLT * y = (FLT *) calloc(n, sizeof(FLT));
    FLT * z = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        x[i] = y[i] = z[i] = 1/(i+ONE(FLT));
    X(samv)('N', alpha, A, x, beta, y);
    X(uamv)('N', R, x);
    X(uamv)('T', R, x);
    for (int i = 0; i < n; i++)
        z[i] = alpha*x[i] + beta*z[i];
    err = X(norm_2arg)(y, z, n)/X(norm_1arg)(y, n);
    printf("Comparison of matrix-vector products \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    for (int i = 0; i < n; i++)
        y[i] = 1/(i+ONE(FLT));
    X(uasv)('T', R, x);
    X(uasv)('N', R, x);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Comparison of matrix-vector solves \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(upper_arrow) * S = X(upper_arrow_inv)(R);
    for (int i = 0; i < n; i++)
        x[i] = y[i];
    X(uamv)('N', S, x);
    X(uamv)('N', R, x);
    X(uamv)('T', S, x);
    X(uamv)('T', R, x);
    X(uasv)('N', S, x);
    X(uasv)('N', R, x);
    X(uasv)('T', S, x);
    X(uasv)('T', R, x);
    err = X(norm_2arg)(x, y, n)/X(norm_1arg)(y, n);
    printf("Comparison of inverse matrix-vector solves \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    X(symmetric_arrow) * B = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    B->a = (FLT *) calloc(n-1, sizeof(FLT));
    B->b = (FLT *) calloc(n-1, sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        B->a[i] = (i+1)*(i+1);
        B->b[i] = i+1;
    }
    B->c = (n*n+n-1);
    B->n = n;
    X(symmetric_arrow) * C = X(symmetric_arrow_congruence)(A, B);
    err = X(pow)(C->c-1, 2);
    for (int i = 0; i < n-1; i++)
        err += X(pow)(C->a[i]-1, 2) + X(pow)(C->b[i]-0, 2);
    err = X(sqrt)(err);
    printf("Self-congruence transformations (Ax = λAx) \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    A = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    A->a = (FLT *) calloc(n-1, sizeof(FLT));
    A->b = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n-1; i++) {
        A->a[i] = (i+1)*(i+1);
        A->b[i] = 1;
        lambda[i] = (i+ONE(FLT)/2)*(i+1);
    }
    A->c = n*n;
    A->n = n;
    lambda[n-1] = n*(n-ONE(FLT)/2);
    X(symmetric_arrow) * AS = X(symmetric_arrow_synthesize)(A, lambda);
    int * p = (int *) malloc(n*sizeof(int));
    int ib = X(symmetric_arrow_deflate)(AS, p);
    free(p);
    FLT * v = (FLT *) calloc(n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        v[i] = (1+X(cbrt)(X(eps)()))*lambda[i];
    FLT * ret = (FLT *) malloc(n*sizeof(FLT));
    FLT * b2 = (FLT *) malloc((n-1)*sizeof(FLT));
    for (int i = 0; i < n-1; i++)
        b2[i] = AS->b[i]*AS->b[i];
    X(secular_FMM)(AS, b2, v, ret, ib);
    FLT * ret_true = (FLT *) calloc(n, sizeof(FLT));
    FLT * cond = (FLT *) calloc(n, sizeof(FLT));
    for (int j = ib+1; j < n-1; j++) {
        ret_true[j] = X(secular)(AS, v[j]);
        cond[j] = X(secular_cond)(AS, v[j]);
    }
    err = X(norm_2arg)(ret+ib+1, ret_true+ib+1, n-2-ib)/X(norm_1arg)(cond+ib+1, n-2-ib);
    X(secular_derivative_FMM)(AS, b2, v, ret, ib);
    for (int j = ib+1; j < n-1; j++) {
        ret_true[j] = X(secular_derivative)(AS, v[j]);
        cond[j] = X(secular_derivative_cond)(AS, v[j]);
    }
    err += X(norm_2arg)(ret+ib+1, ret_true+ib+1, n-2-ib)/X(norm_1arg)(cond+ib+1, n-2-ib);
    X(secular_second_derivative_FMM)(AS, b2, v, ret, ib);
    for (int j = ib+1; j < n-1; j++) {
        ret_true[j] = X(secular_second_derivative)(AS, v[j]);
        cond[j] = X(secular_second_derivative_cond)(AS, v[j]);
    }
    err += X(norm_2arg)(ret+ib+1, ret_true+ib+1, n-2-ib)/X(norm_1arg)(ret_true+ib+1, n-2-ib);
    printf("FMM acceleration of secular(-like) equations \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);

    gettimeofday(&start, NULL);
    FLT * mu = X(symmetric_arrow_eigvals)(AS, ib);
    gettimeofday(&end, NULL);
    err = X(norm_2arg)(lambda, mu, n)/X(norm_1arg)(lambda, n);
    printf("Synthetic symmetric arrow eigenvalues \t\t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);
    printf("Time for symmetric arrow eigenvalues \t\t\t |%20.6f s\n", elapsed(&start, &end, 1));

    gettimeofday(&start, NULL);
    FLT * mu_FMM = X(symmetric_arrow_eigvals_FMM)(AS, ib);
    gettimeofday(&end, NULL);
    err = X(norm_2arg)(lambda, mu_FMM, n)/X(norm_1arg)(lambda, n);
    printf("FMM accelerated symmetric arrow eigenvalues \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);
    printf("Time for FMM accelerated eigenvalues \t\t\t |%20.6f s\n", elapsed(&start, &end, 1));

    X(symmetric_arrow_eigen) * F = X(symmetric_arrow_eig)(A);
    FLT * Q = F->Q;
    FLT * QtQ = (FLT *) calloc(n*n, sizeof(FLT));
    FLT * Id = (FLT *) calloc(n*n, sizeof(FLT));
    for (int i = 0; i < n; i++)
        Id[i+i*n] = 1;
    gettimeofday(&start, NULL);
    for (int j = 0; j < n; j++)
        X(gemv)('T', n, n, 1, Q, Q+j*n, 0, QtQ+j*n);
    gettimeofday(&end, NULL);
    err = X(norm_2arg)(QtQ, Id, n*n)/X(norm_1arg)(Id, n*n);
    free(QtQ);
    printf("Numerical orthogonality of eigenvectors \t\t |%20.2e ", (double) err);
    X(checktest)(err, n, checksum);
    printf("Time to create eigenvectors densely \t\t\t |%20.6f s\n", elapsed(&start, &end, 1));

    X(symmetric_arrow_eigen_FMM) * FFMM = X(symmetric_arrow_eig_FMM)(A);
    FLT * FQ = (FLT *) calloc(n*n, sizeof(FLT));
    QtQ = (FLT *) calloc(n*n, sizeof(FLT));
    gettimeofday(&start, NULL);
    for (int j = 0; j < FFMM->ib; j++)
        FQ[j+j*n] = 1;
    #pragma omp parallel for
    for (int j = FFMM->ib; j < n; j++) {
        X(himv)('N', 1, FFMM->Q, Id+FFMM->ib+j*n, 0, FQ+FFMM->ib+j*n);
        FLT t = 0;
        for (int i = 0; i < n-FFMM->ib; i++)
             t += FFMM->q[i]*Id[i+FFMM->ib+j*n];
        (FQ+j*n)[n-1] = t;
    }
    for (int j = 0; j < FFMM->ib; j++)
        QtQ[j+j*n] = 1;
    #pragma omp parallel for
    for (int j = FFMM->ib; j < n; j++) {
        X(himv)('T', 1, FFMM->Q, FQ+FFMM->ib+j*n, 0, QtQ+FFMM->ib+j*n);
        for (int i = 0; i < n-FFMM->ib; i++)
             QtQ[i+FFMM->ib+j*n] += FFMM->q[i]*FQ[n-1+j*n];
    }
    gettimeofday(&end, NULL);
    //for (int j = 0; j < n; j++)
    //    X(gemv)('T', n, n, 1, FQ, FQ+j*n, 0, QtQ+j*n);
    err = X(norm_2arg)(QtQ, Id, n*n)/X(norm_1arg)(Id, n*n);
    printf("FMM accelerated orthogonality of eigenvectors \t\t |%20.2e ", (double) err);
    X(checktest)(err, n*n, checksum);
    printf("Time to create eigenvectors hierarchically \t\t |%20.6f s\n", elapsed(&start, &end, 1));

    X(destroy_symmetric_arrow)(A);
    X(destroy_symmetric_arrow)(AS);
    X(destroy_symmetric_arrow)(B);
    X(destroy_symmetric_arrow)(C);
    X(destroy_symmetric_arrow_eigen)(F);
    X(destroy_symmetric_arrow_eigen_FMM)(FFMM);
    X(destroy_upper_arrow)(R);
    X(destroy_upper_arrow)(S);
    free(b2);
    free(d_true);
    free(e_true);
    free(x);
    free(y);
    free(z);
    free(lambda);
    free(mu);
    free(mu_FMM);
    free(ret);
    free(ret_true);
    free(cond);
    free(FQ);
    free(QtQ);
    free(Id);
}