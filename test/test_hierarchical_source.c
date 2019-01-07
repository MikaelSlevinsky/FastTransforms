void X(inner_test_hierarchical)(int * checksum, int m, int n, FLT (*f)(FLT x, FLT y), FLT * x, FLT * y) {
    int NLOOPS = 3;
    struct timeval start, end;
    unitrange ir = {0, m}, jr = {0, n};
    FLT err = 0;
    gettimeofday(&start, NULL);
    for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
        X(densematrix) * A = X(sample_densematrix)(f, x, y, ir, jr);
        X(destroy_densematrix)(A);
    }
    gettimeofday(&end, NULL);
    X(densematrix) * A = X(sample_densematrix)(f, x, y, ir, jr);
    printf("Time to sample densely \t\t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

    gettimeofday(&start, NULL);
    for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
        X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(f, x, y, ir, jr);
        X(destroy_hierarchicalmatrix)(H);
    }
    gettimeofday(&end, NULL);
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(f, x, y, ir, jr);
    printf("Time to sample hierarchically \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

    FLT * u = (FLT *) malloc(n*sizeof(FLT));
    for (int j = 0; j < n; j++)
        u[j] = 1/(j+ONE(FLT));
    FLT * v = (FLT *) calloc(m, sizeof(FLT));
    gettimeofday(&start, NULL);
    for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
        X(demv)('N', 1, A, u, 0, v);
    }
    gettimeofday(&end, NULL);
    printf("Time to multiply densely \t\t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

    FLT * w = (FLT *) calloc(m, sizeof(FLT));
    gettimeofday(&start, NULL);
    for (int ntimes = 0; ntimes < NLOOPS; ntimes++) {
        X(himv)('N', 1, H, u, 0, w);
    }
    gettimeofday(&end, NULL);
    printf("Time to multiply hierarchically \t (%5i×%5i) \t |%20.6f s\n", m, n, elapsed(&start, &end, NLOOPS));

    err = X(norm_2arg)(v, w, m)/X(norm_1arg)(v, m);
    printf("Comparison of matrix-vector products \t (%5i×%5i) \t |%20.2e ", m, n, (double) err);
    X(checktest)(err, MAX(m, n), checksum);

    X(destroy_densematrix)(A);
    X(destroy_hierarchicalmatrix)(H);
    free(u);
    free(v);
    free(w);
}

void X(test_hierarchical)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("\t\t\t\t\t\t\t |   or Calculation Time\n");
    printf("---------------------------------------------------------|----------------------\n");

    int nmin = 800, nmax = 3200;
    FLT err = 0;

    for (int n = 40; n < 42; n++) {
        FLT num = 0, den = 0, pt = 0.125;
        FLT fcpt = X(exp)(pt);
        FLT * xc = X(chebyshev_points)('1', n);
        FLT * lc = X(chebyshev_barycentric_weights)('1', n);
        for (int i = 0; i < n; i++) {
            num += X(exp)(xc[i])*lc[i]/(pt-xc[i]);
            den += lc[i]/(pt-xc[i]);
        }
        err += X(fabs)((num/den-fcpt)/fcpt);
        free(xc);
        free(lc);
        xc = X(chebyshev_points)('2', n);
        lc = X(chebyshev_barycentric_weights)('2', n);
        num = den = 0;
        for (int i = 0; i < n; i++) {
            num += X(exp)(xc[i])*lc[i]/(pt-xc[i]);
            den += lc[i]/(pt-xc[i]);
        }
        err += X(fabs)((num/den-fcpt)/fcpt);
        free(xc);
        free(lc);
    }
    printf("Approximation by second-kind barycentric interpolant \t |%20.2e ", (double) err);
    X(checktest)(err, 8, checksum);

    printf("\t\t\t\t\t\t\t |\n");
    printf("The Cauchy kernel and quadratically spaced samples \t |\n");
    printf("\t\t\t\t\t\t\t |\n");
    for (int n = nmin; n < nmax; n *= 2) {
        int m = n;
        FLT * x = (FLT *) malloc(m*sizeof(FLT));
        for (int i = 0; i < m; i++)
            x[i] = (i+1)*(i+1);
        FLT * y = (FLT *) malloc(n*sizeof(FLT));
        for (int j = 0; j < n; j++)
            y[j] = (j+1)*(j+3/TWO(FLT));
        X(inner_test_hierarchical)(checksum, m, n, X(cauchykernel), x, y);
        free(x);
        free(y);
    }
    printf("\t\t\t\t\t\t\t |\n");
    printf("The Coulomb kernel and linearly spaces samples \t\t |\n");
    printf("\t\t\t\t\t\t\t |\n");
    for (int n = nmin; n < nmax; n *= 2) {
        int m = n;
        FLT * x = (FLT *) malloc(m*sizeof(FLT));
        for (int i = 0; i < m; i++)
            x[i] = i+1;
        FLT * y = (FLT *) malloc(n*sizeof(FLT));
        for (int j = 0; j < n; j++)
            y[j] = j+1/TWO(FLT);
        X(inner_test_hierarchical)(checksum, m, n, X(coulombkernel), x, y);
        free(x);
        free(y);
    }
    printf("\t\t\t\t\t\t\t |\n");
    printf("The log kernel and Chebyshev samples \t\t\t |\n");
    printf("\t\t\t\t\t\t\t |\n");
    for (int n = nmin; n < nmax; n *= 2) {
        int m = n;
        FLT * x = X(chebyshev_points)('1', m);
        for (int i = 0; i < m; i++)
            x[i] = -x[i];
        FLT * y = X(chebyshev_points)('2', n);
        for (int j = 0; j < n; j++)
            y[j] = -y[j];
        X(inner_test_hierarchical)(checksum, m, n, X(logkernel), x, y);
        free(x);
        free(y);
    }
}
