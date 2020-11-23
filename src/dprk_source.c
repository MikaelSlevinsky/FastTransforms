void X(destroy_symmetric_dpr1)(X(symmetric_dpr1) * A) {
    free(A->d);
    free(A->z);
    free(A);
}

void X(destroy_symmetric_idpr1)(X(symmetric_idpr1) * A) {
    free(A->z);
    free(A);
}

void X(destroy_symmetric_dpr1_eigen)(X(symmetric_dpr1_eigen) * F) {
    free(F->v);
    free(F->V);
    free(F->lambda);
    free(F->lambdalo);
    free(F->lambdahi);
    free(F->p);
    free(F->q);
    free(F);
}

void X(destroy_symmetric_dpr1_eigen_FMM)(X(symmetric_dpr1_eigen_FMM) * F) {
    X(destroy_symmetric_dpr1)(F->A);
    X(destroy_symmetric_idpr1)(F->B);
    free(F->v);
    X(destroy_hierarchicalmatrix)(F->V);
    free(F->lambda);
    free(F->lambdalo);
    free(F->lambdahi);
    free(F->p);
    free(F->q);
    free(F);
}

size_t X(summary_size_symmetric_dpr1_eigen)(X(symmetric_dpr1_eigen) * F) {
    size_t n = F->n, iz = F->iz, id = F->iz, S = 0;
    S += n*(2*sizeof(int)+3*sizeof(FLT));
    S += id*sizeof(FLT);
    S += sizeof(FLT)*(n-iz)*(n-iz-id);
    return S;
}

size_t X(summary_size_symmetric_dpr1_eigen_FMM)(X(symmetric_dpr1_eigen_FMM) * F) {
    size_t n = F->n, iz = F->iz, id = F->iz, S = 0;
    S += n*(2*sizeof(int)+3*sizeof(FLT));
    S += id*sizeof(FLT);
    S += X(summary_size_hierarchicalmatrix)(F->V);
    return S;
}

X(symmetric_idpr1) * X(symmetric_idpr1_factorize)(X(symmetric_idpr1) * A) {
    int n = A->n;
    FLT * z = A->z;
    FLT sigma = A->sigma;
    FLT * y = malloc(n*sizeof(FLT));
    FLT tau = ZERO(FLT);

    for (int i = 0; i < n; i++) {
        y[i] = z[i];
        tau += z[i]*z[i];
    }
    tau = sigma/(1+Y(sqrt)(1+sigma*tau));

    X(symmetric_idpr1) * B = malloc(sizeof(X(symmetric_idpr1)));
    B->z = y;
    B->sigma = tau;
    B->n = n;
    return B;
}

X(symmetric_idpr1) * X(symmetric_idpr1_inv)(X(symmetric_idpr1) * A) {
    int n = A->n;
    FLT * z = A->z;
    FLT sigma = A->sigma;
    FLT * y = malloc(n*sizeof(FLT));
    FLT tau = ZERO(FLT);

    for (int i = 0; i < n; i++) {
        y[i] = z[i];
        tau += z[i]*z[i];
    }
    tau = -sigma/(1+sigma*tau);

    X(symmetric_idpr1) * B = malloc(sizeof(X(symmetric_idpr1)));
    B->z = y;
    B->sigma = tau;
    B->n = n;
    return B;
}

X(symmetric_dpr1) * X(symmetric_dpr1_inv)(X(symmetric_dpr1) * A) {
    int n = A->n;
    FLT * d = A->d;
    FLT * z = A->z;
    FLT rho = A->rho;
    FLT * e = malloc(n*sizeof(FLT));
    FLT * y = malloc(n*sizeof(FLT));
    FLT sigma = ZERO(FLT);

    for (int i = 0; i < n; i++) {
        e[i] = 1/d[i];
        y[i] = e[i]*z[i];
        sigma += z[i]*y[i];
    }
    sigma = -rho/(1+rho*sigma);

    X(symmetric_dpr1) * B = malloc(sizeof(X(symmetric_dpr1)));
    B->d = e;
    B->z = y;
    B->rho = sigma;
    B->n = n;
    return B;
}

void X(symmetric_dpr1_synthesize)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, zk;
    for (int k = 0; k < n; k++) {
        zk = (X(diff)(lambdahi[k], d[k]) + lambdalo[k])/rho;
        for (int i = 0; i < k; i++)
            zk *= (X(diff)(lambdahi[i], d[k]) + lambdalo[i])/(d[i] - d[k]);
        for (int i = k+1; i < n; i++)
            zk *= (X(diff)(lambdahi[i], d[k]) + lambdalo[i])/(d[i] - d[k]);
        z[k] = Y(copysign)(Y(sqrt)(zk), z[k]);
    }
}

void X(symmetric_definite_dpr1_synthesize)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma, zsum = 0;
    for (int k = 0; k < n; k++) {
        z[k] = (X(diff)(lambdahi[k], d[k]) + lambdalo[k])/(rho - sigma*d[k]);
        for (int i = 0; i < k; i++)
            z[k] *= (X(diff)(lambdahi[i], d[k]) + lambdalo[i])/(d[i] - d[k]);
        for (int i = k+1; i < n; i++)
            z[k] *= (X(diff)(lambdahi[i], d[k]) + lambdalo[i])/(d[i] - d[k]);
        zsum += z[k];
    }
    for (int k = 0; k < n; k++)
        B->z[k] = z[k] = Y(copysign)(Y(sqrt)(z[k]/(1-sigma*zsum)), B->z[k]);
}

// x ← A*x, x ← Aᵀ*x
void X(drmv)(char TRANS, X(symmetric_dpr1) * A, FLT * x) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, ztx = ZERO(FLT);
    if (TRANS == 'N' || TRANS == 'T') {
        for (int i = 0; i < n; i++)
            ztx += z[i]*x[i];
        ztx *= rho;
        for (int i = 0; i < n; i++)
            x[i] = d[i]*x[i] + ztx*z[i];
    }
}

// x ← A*x, x ← Aᵀ*x
void X(irmv)(char TRANS, X(symmetric_idpr1) * A, FLT * x) {
    int n = A->n;
    FLT * z = A->z, sigma = A->sigma, ztx = ZERO(FLT);
    if (TRANS == 'N' || TRANS == 'T') {
        for (int i = 0; i < n; i++)
            ztx += z[i]*x[i];
        ztx *= sigma;
        for (int i = 0; i < n; i++)
            x[i] += ztx*z[i];
    }
}

// y ← α*V*x + β*y, y ← α*Vᵀ*x + β*y
void X(dvmv)(char TRANS, FLT alpha, X(symmetric_dpr1_eigen) * F, FLT * x, FLT beta, FLT * y) {
    int n = F->n, iz = F->iz, id = F->id, * p = F->p, * q = F->q;
    FLT * v = F->v;
    if (TRANS == 'N') {
        X(perm)('T', x, q, n);
        X(perm)('N', y, p, n);
        for (int i = 0; i < iz; i++)
            y[i] = alpha*x[i] + beta*y[i];
        X(gemv)('N', n-iz, n-iz-id, alpha, F->V, n-iz, x+iz+id, beta, y+iz);
        for (int i = iz; i < iz+id; i++)
            y[i] += alpha*v[i-iz]*x[i];
        X(perm)('N', x, q, n);
        X(perm)('T', y, p, n);
    }
    else if (TRANS == 'T') {
        X(perm)('N', x, p, n);
        X(perm)('T', y, q, n);
        for (int i = 0; i < iz; i++)
            y[i] = alpha*x[i] + beta*y[i];
        for (int i = iz; i < iz+id; i++)
            y[i] = alpha*v[i-iz]*x[i] + beta*y[i];
        X(gemv)('T', n-iz, n-iz-id, alpha, F->V, n-iz, x+iz, beta, y+iz+id);
        X(perm)('T', x, p, n);
        X(perm)('N', y, q, n);
    }
}

// y ← α*V*x + β*y, y ← α*Vᵀ*x + β*y
void X(dfmv)(char TRANS, FLT alpha, X(symmetric_dpr1_eigen_FMM) * F, FLT * x, FLT beta, FLT * y) {
    int n = F->n, iz = F->iz, id = F->id, * p = F->p, * q = F->q;
    FLT * v = F->v;
    if (TRANS == 'N') {
        X(perm)('T', x, q, n);
        X(perm)('N', y, p, n);
        for (int i = 0; i < iz; i++)
            y[i] = alpha*x[i] + beta*y[i];
        X(ghmv)('N', alpha, F->V, x+iz+id, beta, y+iz);
        for (int i = iz; i < iz+id; i++)
            y[i] += alpha*v[i-iz]*x[i];
        X(perm)('N', x, q, n);
        X(perm)('T', y, p, n);
    }
    else if (TRANS == 'T') {
        X(perm)('N', x, p, n);
        X(perm)('T', y, q, n);
        for (int i = 0; i < iz; i++)
            y[i] = alpha*x[i] + beta*y[i];
        for (int i = iz; i < iz+id; i++)
            y[i] = alpha*v[i-iz]*x[i] + beta*y[i];
        X(ghmv)('T', alpha, F->V, x+iz, beta, y+iz+id);
        X(perm)('T', x, p, n);
        X(perm)('N', y, q, n);
    }
}

FLT X(secular)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, ret = 1/rho, t;
    for (int i = 0; i < n; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        ret += z[i]*t;
    }
    return ret;
}

FLT X(generalized_secular)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma, ret = 1/(sigma*(X(diff)(rho/sigma, lambdahi) - lambdalo)), t;
    for (int i = 0; i < n; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        ret += z[i]*t;
    }
    return ret;
}

/*
void X(secular_FMM)(X(symmetric_dpr1) * A, FLT * b2, FLT * lambda, FLT * ret, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    for (int j = ib+1; j < n-1; j++)
        ret[j] = lambda[j] - A->c;
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(X(cauchykernel), lambda, a, (unitrange) {ib+1, n-1}, (unitrange) {ib, n-1});
    X(ghmv)('N', -1, H, b2+ib, 1, ret+(ib+1));
    X(destroy_hierarchicalmatrix)(H);
}
*/

FLT X(secular_derivative)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, ret = ZERO(FLT), t;
    for (int i = 0; i < n; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        ret += t*t;
    }
    return ret;
}

FLT X(generalized_secular_derivative)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma, ret = ZERO(FLT), t;
    for (int i = 0; i < n; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        ret += t*t;
    }
    t = sigma*(X(diff)(rho/sigma, lambdahi) - lambdalo);
    return sigma/(t*t)+ret;
}

/*
void X(secular_derivative_FMM)(X(symmetric_dpr1) * A, FLT * b2, FLT * lambda, FLT * ret, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    for (int j = ib+1; j < n-1; j++)
        ret[j] = ONE(FLT);
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(X(coulombkernel), lambda, a, (unitrange) {ib+1, n-1}, (unitrange) {ib, n-1});
    X(ghmv)('N', 1, H, b2+ib, 1, ret+(ib+1));
    X(destroy_hierarchicalmatrix)(H);
}
*/

FLT X(secular_second_derivative)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, ret = ZERO(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        t = t*t/(X(diff)(d[i], lambdahi) - lambdalo);
        ret += t;
    }
    return TWO(FLT)*ret;
}

FLT X(generalized_secular_second_derivative)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma, ret = ZERO(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        t = t*t/(X(diff)(d[i], lambdahi) - lambdalo);
        ret += t;
    }
    t = sigma*(X(diff)(rho/sigma, lambdahi) - lambdalo);
    return TWO(FLT)*(sigma/t*sigma/(t*t)+ret);
}

/*
void X(secular_second_derivative_FMM)(X(symmetric_dpr1) * A, FLT * b2, FLT * lambda, FLT * ret, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    for (int j = ib+1; j < n-1; j++)
        ret[j] = ZERO(FLT);
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(X(coulombprimekernel), lambda, a, (unitrange) {ib+1, n-1}, (unitrange) {ib, n-1});
    X(ghmv)('N', -2, H, b2+ib, 1, ret+(ib+1));
    X(destroy_hierarchicalmatrix)(H);
}
*/

FLT X(exterior_initial_guess)(FLT d0n, FLT nrmz2, FLT rho) {return d0n + rho*nrmz2;}

FLT X(first_pick_zero_update)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z;
    FLT f = X(secular)(A, lambdalo, lambdahi);
    FLT fp = X(secular_derivative)(A, lambdalo, lambdahi);
    FLT d0 = d[0];
    FLT alpha = ONE(FLT), t;
    for (int i = 1; i < n; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        alpha += t*t*(d0-d[i])/(X(diff)(lambdahi, d[i]) + lambdalo);
    }
    FLT a = alpha/(X(diff)(lambdahi, d0) + lambdalo);
    FLT b = fp + f/(X(diff)(lambdahi, d0) + lambdalo);
    FLT c = -f;
    return 2*c/(b+Y(sqrt)(b*b-4*a*c));
}

FLT X(first_generalized_pick_zero_update)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z;
    FLT f = X(generalized_secular)(A, B, lambdalo, lambdahi);
    FLT fp = X(generalized_secular_derivative)(A, B, lambdalo, lambdahi);
    FLT d0 = d[0];
    FLT alpha = ONE(FLT), t;
    for (int i = 1; i < n; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        alpha += t*t*(d0-d[i])/(X(diff)(lambdahi, d[i]) + lambdalo);
    }
    FLT a = alpha/(X(diff)(lambdahi, d0) + lambdalo);
    FLT b = fp + f/(X(diff)(lambdahi, d0) + lambdalo);
    FLT c = -f;
    return 2*c/(b+Y(sqrt)(b*b-4*a*c));
}

FLT X(last_pick_zero_update)(X(symmetric_dpr1) * A, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z;
    FLT f = X(secular)(A, lambdalo, lambdahi);
    FLT fp = X(secular_derivative)(A, lambdalo, lambdahi);
    FLT dn = d[n-1];
    FLT alpha = ONE(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        alpha += t*t*(dn-d[i])/(X(diff)(lambdahi, d[i]) + lambdalo);
    }
    FLT a = alpha/(X(diff)(lambdahi, dn) + lambdalo);
    FLT b = fp + f/(X(diff)(lambdahi, dn) + lambdalo);
    FLT c = -f;
    return 2*c/(b+Y(sqrt)(b*b-4*a*c));
}

FLT X(last_generalized_pick_zero_update)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT lambdalo, FLT lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z;
    FLT f = X(generalized_secular)(A, B, lambdalo, lambdahi);
    FLT fp = X(generalized_secular_derivative)(A, B, lambdalo, lambdahi);
    FLT dn = d[n-1];
    FLT alpha = ONE(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = z[i]/(X(diff)(d[i], lambdahi) - lambdalo);
        alpha += t*t*(dn-d[i])/(X(diff)(lambdahi, d[i]) + lambdalo);
    }
    FLT a = alpha/(X(diff)(lambdahi, dn) + lambdalo);
    FLT b = fp + f/(X(diff)(lambdahi, dn) + lambdalo);
    FLT c = -f;
    return 2*c/(b+Y(sqrt)(b*b-4*a*c));
}

FLT X(pick_zero_update)(X(symmetric_dpr1) * A, FLT x0, FLT x1, FLT lambdalo, FLT lambdahi) {
    FLT f = X(secular)(A, lambdalo, lambdahi);
    FLT fp = X(secular_derivative)(A, lambdalo, lambdahi);
    FLT fpp = X(secular_second_derivative)(A, lambdalo, lambdahi);
    FLT c1 = 1/(X(diff)(x0, lambdahi) - lambdalo);
    FLT c2 = 1/(X(diff)(x1, lambdahi) - lambdalo);
    FLT c2g = (fpp - 2*c1*fp)/(2*(c2-c1)*c2);
    FLT c1b = (fp-c2*c2g)/c1;
    FLT alpha = f - c1b - c2g;
    FLT a = alpha/((X(diff)(x0, lambdahi) - lambdalo)*(X(diff)(lambdahi, x1) + lambdalo));
    FLT b = fp - (c1+c2)*f;
    FLT c = -f;
    return 2*c/(b+Y(sqrt)(b*b-4*a*c));
}

FLT X(generalized_pick_zero_update)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT x0, FLT x1, FLT lambdalo, FLT lambdahi) {
    FLT f = X(generalized_secular)(A, B, lambdalo, lambdahi);
    FLT fp = X(generalized_secular_derivative)(A, B, lambdalo, lambdahi);
    FLT fpp = X(generalized_secular_second_derivative)(A, B, lambdalo, lambdahi);
    FLT c1 = 1/(X(diff)(x0, lambdahi) - lambdalo);
    FLT c2 = 1/(X(diff)(x1, lambdahi) - lambdalo);
    FLT c2g = (fpp - 2*c1*fp)/(2*(c2-c1)*c2);
    FLT c1b = (fp-c2*c2g)/c1;
    FLT alpha = f - c1b - c2g;
    FLT a = alpha/((X(diff)(x0, lambdahi) - lambdalo)*(X(diff)(lambdahi, x1) + lambdalo));
    FLT b = fp - (c1+c2)*f;
    FLT c = -f;
    return 2*c/(b+Y(sqrt)(b*b-4*a*c));
}

/*
void X(pick_zero_update_FMM)(X(symmetric_dpr1) * A, FLT * b2, FLT * lambda, FLT * delta, FLT * f, FLT * fp, FLT * fpp, int ib) {
    int n = A->n;
    FLT * Aa = A->a;
    X(secular_FMM)(A, b2, lambda, f, ib);
    X(secular_derivative_FMM)(A, b2, lambda, fp, ib);
    X(secular_second_derivative_FMM)(A, b2, lambda, fpp, ib);
    FLT c1, c2, c2g, c1b, alpha, a, b, c;
    for (int j = ib+1; j < n-1; j++) {
        c1 = 1/(Aa[j-1] - lambda[j]);
        c2 = 1/(Aa[j] - lambda[j]);
        c2g = (fpp[j] - 2*c1*fp[j])/(2*(c2-c1)*c2);
        c1b = (fp[j]-c2*c2g)/c1;
        alpha = f[j] - c1b - c2g;
        a = alpha/((Aa[j-1] - lambda[j])*(lambda[j] - Aa[j]));
        b = fp[j] - (c1+c2)*f[j];
        c = -f[j];
        delta[j] = 2*c/(b+Y(sqrt)(b*b-4*a*c));
    }
}
*/

// Note: This modifies `A`.
void X(symmetric_dpr1_deflate)(X(symmetric_dpr1) * A, int * p) {
    int n = A->n, iz;
    FLT * d = A->d, * z = A->z, rho = A->rho;
    FLT absrho = Y(fabs)(rho), nrmz = 0;
    for (int i = 0; i < n; i++)
        nrmz += z[i]*z[i];
    nrmz = Y(sqrt)(nrmz);
    // absolute sort based on the rank-one modification `z`.
    X(quicksort_2arg)(z, d, p, 0, n-1, X(ltabs));
    // `|z[iz]|` is the first value that surpasses ` √ |ρ| ||z||_2 × ϵ` in magnitude.
    for (iz = 0; iz < n; iz++)
        if (Y(fabs)(z[iz]) > Y(sqrt)(absrho)*nrmz*Y(eps)()) break;
    // re-sort based on ensuring the elements of `d` with non-negligible
    // corresponding `z` are increasing.
    X(quicksort_2arg)(d, z, p, iz, n-1, X(lt));
    // Deflation based on a Givens rotation to zero-out an entry in `z`.
    for (int i = iz; i < n-1; i++)
        if (Y(fabs)(d[i]-d[i+1]) < MAX(Y(fabs)(d[i]), Y(fabs)(d[i+1]))*Y(eps)())
            printf("Diagonal entries are too close!\n");
    for (int i = iz; i < n; i++) {
        d[i-iz] = d[i];
        z[i-iz] = z[i];
    }
    A->n = n-iz;
}

int X(symmetric_dpr1_deflate2)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi, int * p) {
    int n = A->n, id;
    FLT * d = A->d, * z = A->z;
    // absolute sort based on the λ_{lo}
    X(quicksort_4arg)(lambdalo, lambdahi, d, z, p, 0, n-1, X(ltabs));
    // Find all sufficiently small λ_{lo}
    for (id = 0; id < n; id++)
        if (Y(fabs)(lambdalo[id]) > Y(sqrt)(Y(floatmin)())) break;
    // re-sort based on ensuring the elements of `d` with non-negligible
    // corresponding `z` are increasing.
    X(quicksort_4arg)(d, z, lambdalo, lambdahi, p, id, n-1, X(lt));
    return id;
}

// Note: This modifies `A` and `B`.
void X(symmetric_definite_dpr1_deflate)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, int * p) {
    int n = A->n, iz;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma;
    FLT absrho = Y(fabs)(rho), abssigma = Y(fabs)(sigma), nrmz = 0;
    for (int i = 0; i < n; i++)
        nrmz += z[i]*z[i];
    nrmz = Y(sqrt)(nrmz);
    // absolute sort based on the rank-one modification `z`.
    X(quicksort_2arg)(z, d, p, 0, n-1, X(ltabs));
    // `|z[iz]|` is the first value that surpasses ` √ (|ρ|+|σ|) ||z||_2 × ϵ` in magnitude.
    for (iz = 0; iz < n; iz++)
        if (Y(fabs)(z[iz]) > Y(sqrt)(absrho+abssigma)*nrmz*Y(eps)()) break;
    // re-sort based on ensuring the elements of `d` with non-negligible
    // corresponding `z` are increasing.
    X(quicksort_2arg)(d, z, p, iz, n-1, X(lt));
    // Deflation based on a Givens rotation to zero-out an entry in `z`.
    for (int i = iz; i < n-1; i++)
        if (Y(fabs)(d[i]-d[i+1]) < MAX(Y(fabs)(d[i]), Y(fabs)(d[i+1]))*Y(eps)())
            printf("Diagonal entries are too close!\n");
    // Deflation based on removal of an entry from `d`.
    for (int i = 0; i < n; i++)
        if (Y(fabs)(d[i]-rho/sigma) < MAX(Y(fabs)(d[i]), absrho/abssigma)*Y(eps)())
            printf("One diagonal entry is too close to ρ/σ!\n");
    for (int i = iz; i < n; i++) {
        d[i-iz] = d[i];
        B->z[i-iz] = z[i-iz] = z[i];
    }
    A->n = B->n = n-iz;
}

int X(symmetric_definite_dpr1_deflate2)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi, int * p) {
    int n = A->n, id;
    FLT * d = A->d, * z = A->z;
    // absolute sort based on the λ_{lo}
    X(quicksort_4arg)(lambdalo, lambdahi, d, z, p, 0, n-1, X(ltabs));
    // Find all sufficiently small λ_{lo}
    for (id = 0; id < n; id++)
        if (Y(fabs)(lambdalo[id]) > Y(sqrt)(Y(floatmin)())) break;
    // re-sort based on ensuring the elements of `d` with non-negligible
    // corresponding `z` are increasing.
    X(quicksort_4arg)(d, z, lambdalo, lambdahi, p, id, n-1, X(lt));
    for (int i = 0; i < n; i++)
        B->z[i] = z[i];
    return id;
}

// Assuming that X(symmetric_dpr1_deflate)(A, p); has been called.
void X(symmetric_dpr1_eigvals)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, nrmz2 = 0;
    for (int i = 0; i < n; i++)
        nrmz2 += z[i]*z[i];
    if (rho == 0)
        for (int i = 0; i < n; i++) {
            lambdahi[i] = d[i];
            lambdalo[i] = 0;
        }
    else if (rho > 0) {
        for (int j = 0; j < n-1; j++) {
            FLT x0 = d[j];
            FLT x1 = d[j+1];
            FLT lambdak = (x0+x1)/2;
            lambdahi[j] = X(secular)(A, 0, lambdak) > 0 ? x0 : x1;
            lambdak = lambdak - lambdahi[j];
            FLT deltak = 1+n*Y(fabs)(lambdak);
            while (Y(fabs)(deltak) > MAX(2*n*Y(fabs)(lambdak)*Y(eps)(), Y(floatmin)())) {
                deltak = X(pick_zero_update)(A, x0, x1, lambdak, lambdahi[j]);
                if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
            }
            deltak = X(pick_zero_update)(A, x0, x1, lambdak, lambdahi[j]);
            if (Y(isfinite)(deltak)) lambdak += deltak;
            lambdalo[j] = lambdak;
        }
        FLT lambdak = X(exterior_initial_guess)(d[n-1], nrmz2, rho);
        lambdahi[n-1] = d[n-1];
        lambdak = lambdak - lambdahi[n-1];
        FLT deltak = 1+n*Y(fabs)(lambdak);
        while (Y(fabs)(deltak) > MAX(2*n*Y(fabs)(lambdak)*Y(eps)(), Y(floatmin)())) {
            deltak = X(last_pick_zero_update)(A, lambdak, lambdahi[n-1]);
            if (Y(isfinite)(deltak)) lambdak += deltak;
            else break;
        }
        deltak = X(last_pick_zero_update)(A, lambdak, lambdahi[n-1]);
        if (Y(isfinite)(deltak)) lambdak += deltak;
        lambdalo[n-1] = lambdak;
    }
    else {
        FLT lambdak = X(exterior_initial_guess)(d[0], nrmz2, rho);
        lambdahi[0] = d[0];
        lambdak = lambdak - lambdahi[0];
        FLT deltak = 1+n*Y(fabs)(lambdak);
        while (Y(fabs)(deltak) > MAX(2*n*Y(fabs)(lambdak)*Y(eps)(), Y(floatmin)())) {
            deltak = X(first_pick_zero_update)(A, lambdak, lambdahi[0]);
            if (Y(isfinite)(deltak)) lambdak += deltak;
            else break;
        }
        deltak = X(first_pick_zero_update)(A, lambdak, lambdahi[0]);
        if (Y(isfinite)(deltak)) lambdak += deltak;
        lambdalo[0] = lambdak;
        for (int j = 1; j < n; j++) {
            FLT x0 = d[j-1];
            FLT x1 = d[j];
            FLT lambdak = (x0+x1)/2;
            lambdahi[j] = X(secular)(A, 0, lambdak) < 0 ? x0 : x1;
            lambdak = lambdak - lambdahi[j];
            FLT deltak = 1+n*Y(fabs)(lambdak);
            while (Y(fabs)(deltak) > MAX(2*n*Y(fabs)(lambdak)*Y(eps)(), Y(floatmin)())) {
                deltak = X(pick_zero_update)(A, x0, x1, lambdak, lambdahi[j]);
                if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
            }
            deltak = X(pick_zero_update)(A, x0, x1, lambdak, lambdahi[j]);
            if (Y(isfinite)(deltak)) lambdak += deltak;
            lambdalo[j] = lambdak;
        }
    }
}

// Assuming that X(symmetric_definite_dpr1_deflate)(A, B, p); has been called.
void X(symmetric_definite_dpr1_eigvals)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi) {
    int i, n = A->n;
    FLT * d = A->d, * z = A->z, rho = A->rho, sigma = B->sigma;
    FLT absrho = Y(fabs)(rho), abssigma = Y(fabs)(sigma), tau = rho/sigma;
    if (sigma == 0)
        X(symmetric_dpr1_eigvals)(A, lambdalo, lambdahi);
    else if (sigma > 0) {
        // Monotonicity preserved.
        for (i = 0; i < n; i++)
            if (d[i] > tau) break;
        i -= 1;
        for (int j = 0; j < n; j++) {
            FLT x0, x1, lambdak, deltak;
            if (j < i) {
                x0 = d[j];
                x1 = d[j+1];
            }
            else if (j == i) {
                x0 = d[j];
                x1 = tau;
            }
            else if (j == i + 1) {
                x0 = tau;
                x1 = d[j];
            }
            else {
                x0 = d[j-1];
                x1 = d[j];
            }
            lambdak = (x0+x1)/2;
            if (lambdak == x0 || lambdak == x1) {
                lambdahi[j] = lambdak;
                lambdalo[j] = 0;
            }
            else {
                lambdahi[j] = X(generalized_secular)(A, B, 0, lambdak) > 0 ? x0 : x1;
                lambdak = lambdak - lambdahi[j];
                deltak = 1+n*Y(fabs)(lambdak);
                while (Y(fabs)(deltak) > MAX(2*n*Y(fabs)(lambdak)*Y(eps)(), Y(floatmin)())) {
                    deltak = X(generalized_pick_zero_update)(A, B, x0, x1, lambdak, lambdahi[j]);
                    if (Y(isfinite)(deltak)) lambdak += deltak;
                    else break;
                }
                deltak = X(generalized_pick_zero_update)(A, B, x0, x1, lambdak, lambdahi[j]);
                if (Y(isfinite)(deltak)) lambdak += deltak;
                lambdalo[j] = lambdak;
            }
        }
    }
    else {
        // Monotonicity broken. Guard iterations with bisection.
        printf("σ < 0 ⇒ monotonicity is broken.\n");
        printf("Please restate your problem so that σ ≥ 0.\n");
        /*
        FLT nrmz2 = 0;
        for (int i = 0; i < n; i++)
            nrmz2 += z[i]*z[i];
        for (i = 0; i < n; i++)
            if (d[i] > tau) break;
        i -= 1;
        // j == 0;
        if (0 <= i) {
            int kx0 = 1, kx1 = 1;
            int kden = kx0+kx1;
            int k = 1;
            FLT x0 = X(exterior_initial_guess)(d[0], nrmz2, -Y(fabs)(rho));
            while (X(generalized_secular)(A, B, x0) > 0) {
                //printf("This is f(x_0): %17.16e.\n", X(generalized_secular)(A, B, x0));
                //printf("No: Problem!\n");
                k += 1;
                x0 = X(exterior_initial_guess)(d[0], nrmz2, -k*Y(fabs)(rho));
            }
            FLT x1 = d[0];
            FLT lambdak = x0;//(x0+x1)/2;
            FLT deltak = ONE(FLT);
            while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
                deltak = X(first_generalized_pick_zero_update)(A, B, lambdak);
                if ((x0 > lambdak+deltak) || (lambdak+deltak > x1)) {
                    if (X(generalized_secular)(A, B, lambdak) > 0) {
                        kx0 += kden;
                        kden = kx0+kx1;
                    }
                    else {
                        kx1 += kden;
                        kden = kx0+kx1;
                    }
                    lambdak = (kx0*x0+kx1*x1)/kden;
                }
                else if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
            }
            lambda[0] = lambdak;
        }
        else {
            int kx0 = 1, kx1 = 1;
            int kden = kx0+kx1;
            FLT x0 = d[0];
            FLT x1 = d[1];
            FLT lambdak = (x0+x1)/2;
            FLT deltak = ONE(FLT);
            while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
                deltak = X(generalized_pick_zero_update)(A, B, x0, x1, lambdak);
                if ((x0 > lambdak+deltak) || (lambdak+deltak > x1)) {
                    if (X(generalized_secular)(A, B, lambdak) > 0) {
                        kx0 += kden;
                        kden = kx0+kx1;
                    }
                    else {
                        kx1 += kden;
                        kden = kx0+kx1;
                    }
                    lambdak = (kx0*x0+kx1*x1)/kden;
                }
                else if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
            }
            lambda[0] = lambdak;
        }
        // 1 ≤ j < n-1;
        for (int j = 1; j < n-1; j++) {
            int kx0 = 1, kx1 = 1;
            int kden = kx0+kx1;
            FLT x0;
            FLT x1;
            if (j <= i) {
                x0 = d[j-1];
                x1 = d[j];
            }
            else {
                x0 = d[j];
                x1 = d[j+1];
            }
            FLT lambdak = (x0+x1)/2;
            FLT deltak = ONE(FLT);
            while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
                deltak = X(generalized_pick_zero_update)(A, B, x0, x1, lambdak);
                if ((x0 > lambdak+deltak) || (lambdak+deltak > x1)) {
                    if (X(generalized_secular)(A, B, lambdak) > 0) {
                        kx0 += kden;
                        kden = kx0+kx1;
                    }
                    else {
                        kx1 += kden;
                        kden = kx0+kx1;
                    }
                    lambdak = (kx0*x0+kx1*x1)/kden;
                }
                else if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
                //printf("This is j: %i, d_{j-1} < λ_k < d_j: %17.16e < %17.16e < %17.16e, and f(λ_k): %17.16e.\n", j, d[j-1], lambdak, d[j], X(generalized_secular)(A, B, lambdak));
            }
            lambda[j] = lambdak;
        }
        // j == n-1;
        if (n-1 <= i) {
            int kx0 = 1, kx1 = 1;
            int kden = kx0+kx1;
            FLT x0 = d[n-2];
            FLT x1 = d[n-1];
            FLT lambdak = (x0+x1)/2;
            FLT deltak = ONE(FLT);
            while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
                deltak = X(generalized_pick_zero_update)(A, B, x0, x1, lambdak);
                if ((x0 > lambdak+deltak) || (lambdak+deltak > x1)) {
                    if (X(generalized_secular)(A, B, lambdak) > 0) {
                        kx0 += kden;
                        kden = kx0+kx1;
                    }
                    else {
                        kx1 += kden;
                        kden = kx0+kx1;
                    }
                    lambdak = (kx0*x0+kx1*x1)/kden;
                }
                else if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
            }
            lambda[n-1] = lambdak;
        }
        else {
            // Create last initial guess.
            int kx0 = 1, kx1 = 1;
            int kden = kx0+kx1;
            int k = 1;
            FLT x0 = d[n-1];
            FLT x1 = X(exterior_initial_guess)(d[n-1], nrmz2, Y(fabs)(rho));
            while (X(generalized_secular)(A, B, x1) < 0) {
                //printf("This is f(x_1): %17.16e.\n", X(generalized_secular)(A, B, x1));
                //printf("No: Problem!\n");
                k += 1;
                x1 = X(exterior_initial_guess)(d[n-1], nrmz2, k*Y(fabs)(rho));
            }
            //printf("This is f(x_1): %17.16e.\n", X(generalized_secular)(A, B, x1));
            // Iterate.
            FLT lambdak = x1;//(x0+x1)/2;
            FLT deltak = ONE(FLT);
            while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
                deltak = X(last_generalized_pick_zero_update)(A, B, lambdak);
                //printf("This is deltak: %17.16e.\n", deltak);
                if ((x0 > lambdak+deltak) || (lambdak+deltak > x1)) {
                    if (X(generalized_secular)(A, B, lambdak) > 0) {
                        kx0 += kden;
                        kden = kx0+kx1;
                    }
                    else {
                        kx1 += kden;
                        kden = kx0+kx1;
                    }
                    lambdak = (kx0*x0+kx1*x1)/kden;
                }
                else if (Y(isfinite)(deltak)) lambdak += deltak;
                else break;
            }
            lambda[n-1] = lambdak;
        }
        */
    }
}



/*
// Assuming that X(symmetric_dpr1_deflate)(A, p); has been called.
FLT * X(symmetric_dpr1_eigvals_FMM)(X(symmetric_dpr1) * A, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, c = A->c;

    FLT * lambda = calloc(n, sizeof(FLT));
    FLT * b2 = malloc((n-1)*sizeof(FLT));

    FLT nrmb2 = 0;
    for (int i = 0; i < n-1; i++) {
        b2[i] = b[i]*b[i];
        nrmb2 += b2[i];
    }

    for (int i = 0; i < ib; i++)
        lambda[i] = a[i];

    FLT lambdak, deltak;

    lambdak = X(first_initial_guess)(a[ib], nrmb2, c);
    deltak = ONE(FLT);
    while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
        deltak = X(first_pick_zero_update)(A, lambdak, ib);
        if (Y(isfinite)(deltak)) lambdak += deltak;
        else break;
    }
    lambda[ib] = lambdak;

    FLT nrmd = 1, nrml = 1;
    FLT * delta = malloc((n-1)*sizeof(FLT));
    FLT * f = malloc(n*sizeof(FLT));
    FLT * fp = malloc(n*sizeof(FLT));
    FLT * fpp = malloc(n*sizeof(FLT));
    for (int j = ib+1; j < n-1; j++)
        lambda[j] = (a[j]+a[j-1])/2;
    while (nrmd > MAX(n*Y(eps)()*nrml, Y(floatmin)())) {
        X(pick_zero_update_FMM)(A, b2, lambda, delta, f, fp, fpp, ib);
        nrmd = nrml = 0;
        for (int j = ib+1; j < n-1; j++) {
            if (Y(isfinite)(delta[j])) {
                lambda[j] += delta[j];
                nrmd += delta[j]*delta[j];
                nrml += lambda[j]*lambda[j];
            }
            else continue;
        }
        nrmd = Y(sqrt)(nrmd);
        nrml = Y(sqrt)(nrml);
    }
    free(b2);
    free(delta);
    free(f);
    free(fp);
    free(fpp);

    lambdak = X(last_initial_guess)(a[n-2], nrmb2, c);
    deltak = ONE(FLT);
    while (Y(fabs)(deltak) > MAX(n*Y(eps)()*Y(fabs)(lambdak), Y(floatmin)())) {
        deltak = X(last_pick_zero_update)(A, lambdak);
        if (Y(isfinite)(deltak)) lambdak += deltak;
        else break;
    }
    lambda[n-1] = lambdak;

    return lambda;
}
*/

// Assumes d_i ≠ λ_j and z_i ≠ 0.
FLT * X(symmetric_dpr1_eigvecs)(X(symmetric_dpr1) * A, FLT * lambdalo, FLT * lambdahi, int m) {
    int n = A->n, k;
    FLT * d = A->d, * z = A->z;
    FLT * Q = calloc(n*m, sizeof(FLT));
    FLT nrm;
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++)
            Q[i+j*n] = z[i]/(X(diff)(d[i], lambdahi[j]) - lambdalo[j]);
        nrm = 0;
        for (int i = 0; i < n; i++)
            nrm += Q[i+j*n]*Q[i+j*n];
        nrm = Y(copysign)(1/Y(sqrt)(nrm), Q[j+j*n]);
        for (int i = 0; i < n; i++)
            Q[i+j*n] *= nrm;
        /*
        k = -1;
        nrm = 0;
        for (int i = 0; i < n; i++) {
            if (Y(isfinite)(Q[i+j*n])) nrm += Q[i+j*n]*Q[i+j*n];
            else {k = i; break;}
        }
        if (k != -1) {
            for (int i = 0; i < n; i++)
                Q[i+j*n] = 0;
            Q[k+j*n] = 1;
        }
        else {
            nrm = Y(copysign)(1/Y(sqrt)(nrm), Q[j+j*n]);
            for (int i = 0; i < n; i++)
                Q[i+j*n] *= nrm;
        }
        */
    }
    return Q;
}

// Assumes d_i ≠ λ_j and z_i ≠ 0.
X(hierarchicalmatrix) * X(symmetric_dpr1_eigvecs_FMM)(X(symmetric_dpr1) * A, FLT * lambda, FLT * lambdalo, FLT * lambdahi, int m) {
    int n = A->n;
    FLT * d = A->d, * z = A->z;
    X(hierarchicalmatrix) * Q = X(sample_accurately_hierarchicalmatrix)(X(cauchykernel), X(cauchykernel2), d, lambda, lambdalo, lambdahi, (unitrange) {0, n}, (unitrange) {0, m}, 'G');
    X(hierarchicalmatrix) * N = X(sample_accurately_hierarchicalmatrix)(X(coulombkernel), X(coulombkernel2), d, lambda, lambdalo, lambdahi, (unitrange) {0, n}, (unitrange) {0, m}, 'G');
    FLT * q = calloc(m, sizeof(FLT));
    X(scale_rows_hierarchicalmatrix)(1, z, N);
    X(ghmv)('T', 1, N, z, 0, q);
    for (int j = 0; j < m; j++)
        q[j] = Y(sqrt)(1/q[j]);
    X(scale_rows_hierarchicalmatrix)(1, z, Q);
    X(scale_columns_hierarchicalmatrix)(1, q, Q);
    X(destroy_hierarchicalmatrix)(N);
    free(q);
    return Q;
}

// Assumes d_i ≠ λ_j and z_i ≠ 0.
FLT * X(symmetric_definite_dpr1_eigvecs)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambdalo, FLT * lambdahi, int m) {
    int n = A->n, k;
    FLT * d = A->d, * z = A->z, sigma = B->sigma;
    FLT * V = calloc(n*m, sizeof(FLT));
    FLT nrm, nrm1, nrm2;
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++)
            V[i+j*n] = z[i]/(X(diff)(d[i], lambdahi[j]) - lambdalo[j]);
        nrm1 = nrm2 = 0;
        for (int i = 0; i < n; i++) {
            nrm1 += V[i+j*n]*V[i+j*n];
            nrm2 += V[i+j*n]*z[i];
        }
        nrm = Y(copysign)(1/Y(sqrt)(nrm1+sigma*nrm2*nrm2), V[j+j*n]);
        for (int i = 0; i < n; i++)
            V[i+j*n] *= nrm;
        /*
        k = -1;
        nrm1 = nrm2 = 0;
        for (int i = 0; i < n; i++) {
            if (Y(isfinite)(V[i+j*n])) {
                nrm1 += V[i+j*n]*V[i+j*n];
                nrm2 += V[i+j*n]*z[i];
            }
            else {k = i; break;}
        }
        if (k != -1) {
            for (int i = 0; i < n; i++)
                V[i+j*n] = 0;
            V[k+j*n] = Y(sqrt)(1/(1+sigma*z[k]*z[k]));
        }
        else {
            nrm = Y(copysign)(1/Y(sqrt)(nrm1+sigma*nrm2*nrm2), V[j+j*n]);
            for (int i = 0; i < n; i++)
                V[i+j*n] *= nrm;
        }
        */
    }
    return V;
}

// Assumes d_i ≠ λ_j and z_i ≠ 0.
X(hierarchicalmatrix) * X(symmetric_definite_dpr1_eigvecs_FMM)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B, FLT * lambda, FLT * lambdalo, FLT * lambdahi, int m) {
    int n = A->n;
    FLT * d = A->d, * z = A->z, sigma = B->sigma;
    X(hierarchicalmatrix) * V = X(sample_accurately_hierarchicalmatrix)(X(cauchykernel), X(cauchykernel2), d, lambda, lambdalo, lambdahi, (unitrange) {0, n}, (unitrange) {0, m}, 'G');
    X(hierarchicalmatrix) * N = X(sample_accurately_hierarchicalmatrix)(X(coulombkernel), X(coulombkernel2), d, lambda, lambdalo, lambdahi, (unitrange) {0, n}, (unitrange) {0, m}, 'G');
    FLT * v = calloc(m, sizeof(FLT));
    X(scale_rows_hierarchicalmatrix)(1, z, V);
    X(scale_rows_hierarchicalmatrix)(1, z, N);
    X(ghmv)('T', 1, V, z, 0, v);
    for (int j = 0; j < m; j++)
        v[j] *= v[j];
    X(ghmv)('T', 1, N, z, sigma, v);
    for (int j = 0; j < m; j++)
        v[j] = Y(sqrt)(1/v[j]);
    X(scale_columns_hierarchicalmatrix)(1, v, V);
    X(destroy_hierarchicalmatrix)(N);
    free(v);
    return V;
}

// Note: this modifies `A`.
X(symmetric_dpr1_eigen) * X(symmetric_dpr1_eig)(X(symmetric_dpr1) * A) {
    int n = A->n;
    FLT * lambdalo = calloc(n, sizeof(FLT));
    FLT * lambdahi = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambdahi[i] = A->d[i];
    int * p1 = malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        p1[i] = i;
    X(symmetric_dpr1_deflate)(A, p1);
    X(perm)('N', lambdalo, p1, n);
    X(perm)('N', lambdahi, p1, n);
    int iz = n-A->n;
    X(symmetric_dpr1_eigvals)(A, lambdalo+iz, lambdahi+iz);
    int * p2 = malloc((n-iz)*sizeof(int));
    for (int i = 0; i < n-iz; i++)
        p2[i] = i;
    int id = X(symmetric_dpr1_deflate2)(A, lambdalo+iz, lambdahi+iz, p2);
    FLT * v = malloc(id*sizeof(FLT));
    for (int i = 0; i < id; i++)
        v[i] = 1;
    int * p = malloc(n*sizeof(int));
    for (int i = 0; i < iz; i++)
        p[i] = p1[i];
    for (int i = iz; i < n; i++)
        p[i] = p1[p2[i-iz]+iz];
    free(p1);
    free(p2);

    int * q = malloc(n*sizeof(int));
    FLT * lambda = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++) {
        q[i] = i;
        lambda[i] = lambdahi[i]+lambdalo[i];
    }
    FLT * Q = X(symmetric_dpr1_eigvecs)(A, lambdalo+iz, lambdahi+iz, n-iz-id);
    X(quicksort_3arg)(lambda, lambdalo, lambdahi, q, 0, n-1, X(lt));

    X(symmetric_dpr1_eigen) * F = malloc(sizeof(X(symmetric_dpr1_eigen)));
    F->v = v;
    F->V = Q;
    F->lambda = lambda;
    F->lambdalo = lambdalo;
    F->lambdahi = lambdahi;
    F->p = p;
    F->q = q;
    F->n = n;
    F->iz = iz;
    F->id = id;
    return F;
}

// Note: this modifies `A`.
X(symmetric_dpr1_eigen_FMM) * X(symmetric_dpr1_eig_FMM)(X(symmetric_dpr1) * A) {
    int n = A->n;
    FLT * lambdalo = calloc(n, sizeof(FLT));
    FLT * lambdahi = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambdahi[i] = A->d[i];
    int * p1 = malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        p1[i] = i;
    X(symmetric_dpr1_deflate)(A, p1);
    X(perm)('N', lambdalo, p1, n);
    X(perm)('N', lambdahi, p1, n);
    int iz = n-A->n;
    X(symmetric_dpr1_eigvals)(A, lambdalo+iz, lambdahi+iz);
    int * p2 = malloc((n-iz)*sizeof(int));
    for (int i = 0; i < n-iz; i++)
        p2[i] = i;
    int id = X(symmetric_dpr1_deflate2)(A, lambdalo+iz, lambdahi+iz, p2);
    FLT * v = malloc(id*sizeof(FLT));
    for (int i = 0; i < id; i++)
        v[i] = 1;
    int * p = malloc(n*sizeof(int));
    for (int i = 0; i < iz; i++)
        p[i] = p1[i];
    for (int i = iz; i < n; i++)
        p[i] = p1[p2[i-iz]+iz];
    free(p1);
    free(p2);

    int * q = malloc(n*sizeof(int));
    FLT * lambda = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++) {
        q[i] = i;
        lambda[i] = lambdahi[i]+lambdalo[i];
    }
    X(hierarchicalmatrix) * Q = X(symmetric_dpr1_eigvecs_FMM)(A, lambda+iz, lambdalo+iz, lambdahi+iz, n-iz-id);
    X(quicksort_3arg)(lambda, lambdalo, lambdahi, q, 0, n-1, X(lt));

    X(symmetric_dpr1) * FA = malloc(sizeof(X(symmetric_dpr1)));
    X(symmetric_idpr1) * FB = malloc(sizeof(X(symmetric_idpr1)));
    FB->n = FA->n = A->n;
    FA->d = malloc(FA->n*sizeof(FLT));
    FA->z = malloc(FA->n*sizeof(FLT));
    FB->z = malloc(FA->n*sizeof(FLT));
    for (int i = 0; i < FA->n; i++) {
        FA->d[i] = A->d[i];
        FB->z[i] = FA->z[i] = A->z[i];
    }
    FA->rho = A->rho;
    FB->sigma = 0;

    X(symmetric_dpr1_eigen_FMM) * F = malloc(sizeof(X(symmetric_dpr1_eigen_FMM)));
    F->A = FA;
    F->B = FB;
    F->v = v;
    F->V = Q;
    F->lambda = lambda;
    F->lambdalo = lambdalo;
    F->lambdahi = lambdahi;
    F->p = p;
    F->q = q;
    F->n = n;
    F->iz = iz;
    F->id = id;
    return F;
}

// Note: this modifies `A` and `B`.
X(symmetric_dpr1_eigen) * X(symmetric_definite_dpr1_eig)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B) {
    int n = A->n;
    FLT * lambdalo = calloc(n, sizeof(FLT));
    FLT * lambdahi = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambdahi[i] = A->d[i];
    int * p1 = malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        p1[i] = i;
    X(symmetric_definite_dpr1_deflate)(A, B, p1);
    X(perm)('N', lambdalo, p1, n);
    X(perm)('N', lambdahi, p1, n);
    int iz = n-A->n;
    X(symmetric_definite_dpr1_eigvals)(A, B, lambdalo+iz, lambdahi+iz);
    int * p2 = malloc((n-iz)*sizeof(int));
    for (int i = 0; i < n-iz; i++)
        p2[i] = i;
    int id = X(symmetric_definite_dpr1_deflate2)(A, B, lambdalo+iz, lambdahi+iz, p2);
    FLT * v = malloc(id*sizeof(FLT));
    for (int i = 0; i < id; i++)
        v[i] = Y(sqrt)(1/(1+B->sigma*B->z[i]*B->z[i]));
    int * p = malloc(n*sizeof(int));
    for (int i = 0; i < iz; i++)
        p[i] = p1[i];
    for (int i = iz; i < n; i++)
        p[i] = p1[p2[i-iz]+iz];
    free(p1);
    free(p2);

    int * q = malloc(n*sizeof(int));
    FLT * lambda = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++) {
        q[i] = i;
        lambda[i] = lambdahi[i]+lambdalo[i];
    }
    FLT * V = X(symmetric_definite_dpr1_eigvecs)(A, B, lambdalo+iz+id, lambdahi+iz+id, n-iz-id);
    X(quicksort_3arg)(lambda, lambdalo, lambdahi, q, 0, n-1, X(lt));

    X(symmetric_dpr1_eigen) * F = malloc(sizeof(X(symmetric_dpr1_eigen)));
    F->v = v;
    F->V = V;
    F->lambda = lambda;
    F->lambdalo = lambdalo;
    F->lambdahi = lambdahi;
    F->p = p;
    F->q = q;
    F->n = n;
    F->iz = iz;
    F->id = id;
    return F;
}

// Note: this modifies `A` and `B`.
X(symmetric_dpr1_eigen_FMM) * X(symmetric_definite_dpr1_eig_FMM)(X(symmetric_dpr1) * A, X(symmetric_idpr1) * B) {
    int n = A->n;
    FLT * lambdalo = calloc(n, sizeof(FLT));
    FLT * lambdahi = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        lambdahi[i] = A->d[i];
    int * p1 = malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        p1[i] = i;
    X(symmetric_definite_dpr1_deflate)(A, B, p1);
    X(perm)('N', lambdalo, p1, n);
    X(perm)('N', lambdahi, p1, n);
    int iz = n-A->n;
    X(symmetric_definite_dpr1_eigvals)(A, B, lambdalo+iz, lambdahi+iz);
    int * p2 = malloc((n-iz)*sizeof(int));
    for (int i = 0; i < n-iz; i++)
        p2[i] = i;
    int id = X(symmetric_definite_dpr1_deflate2)(A, B, lambdalo+iz, lambdahi+iz, p2);
    FLT * v = malloc(id*sizeof(FLT));
    for (int i = 0; i < id; i++)
        v[i] = Y(sqrt)(1/(1+B->sigma*B->z[i]*B->z[i]));
    int * p = malloc(n*sizeof(int));
    for (int i = 0; i < iz; i++)
        p[i] = p1[i];
    for (int i = iz; i < n; i++)
        p[i] = p1[p2[i-iz]+iz];
    free(p1);
    free(p2);

    int * q = malloc(n*sizeof(int));
    FLT * lambda = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++) {
        q[i] = i;
        lambda[i] = lambdahi[i]+lambdalo[i];
    }
    X(hierarchicalmatrix) * V = X(symmetric_definite_dpr1_eigvecs_FMM)(A, B, lambda+iz+id, lambdalo+iz+id, lambdahi+iz+id, n-iz-id);
    X(quicksort_3arg)(lambda, lambdalo, lambdahi, q, 0, n-1, X(lt));

    X(symmetric_dpr1) * FA = malloc(sizeof(X(symmetric_dpr1)));
    X(symmetric_idpr1) * FB = malloc(sizeof(X(symmetric_idpr1)));
    FB->n = FA->n = A->n;
    FA->d = malloc(FA->n*sizeof(FLT));
    FA->z = malloc(FA->n*sizeof(FLT));
    FB->z = malloc(FA->n*sizeof(FLT));
    for (int i = 0; i < FA->n; i++) {
        FA->d[i] = A->d[i];
        FB->z[i] = FA->z[i] = A->z[i];
    }
    FA->rho = A->rho;
    FB->sigma = B->sigma;

    X(symmetric_dpr1_eigen_FMM) * F = malloc(sizeof(X(symmetric_dpr1_eigen_FMM)));
    F->A = FA;
    F->B = FB;
    F->v = v;
    F->V = V;
    F->lambda = lambda;
    F->lambdalo = lambdalo;
    F->lambdahi = lambdahi;
    F->p = p;
    F->q = q;
    F->n = n;
    F->iz = iz;
    F->id = id;
    return F;
}


// No transpose: x .= x[p], or x .= P*x where P = Id[p, :].
// Transpose:    x[p] .= x, or x .= P'x where P = Id[p, :].
void X(perm)(char TRANS, FLT * x, int * p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = p[i]-n;
    if (TRANS == 'N') {
        int j, k;
        for (int i = 0; i < n; i++) {
            if (p[i] >= 0) continue;
            j = i;
            k = p[j] = p[j]+n;
            while (p[k] < 0) {
                X(swap)(x, j, k);
                j = k;
                k = p[j] = p[j]+n;
            }
        }
    }
    else if (TRANS == 'T') {
        int j;
        for (int i = 0; i < n; i++) {
            if (p[i] >= 0) continue;
            j = p[i] = p[i]+n;
            while (p[j] < 0) {
                X(swap)(x, i, j);
                j = p[j] = p[j]+n;
            }
        }
    }
}


/*
These versions of quicksort sort `a` in-place according to the `by` ordering on
`FLT` types. They also return the permutation `p`.
*/

static FLT X(selectpivot_1arg)(FLT * a, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int mid = (lo+hi)/2;
    if (by(a[mid], a[lo])) {
        X(swap)(a, lo, mid);
        X(swapi)(p, lo, mid);
    }
    if (by(a[hi], a[lo])) {
        X(swap)(a, lo, hi);
        X(swapi)(p, lo, hi);
    }
    if (by(a[mid], a[hi])) {
        X(swap)(a, mid, hi);
        X(swapi)(p, mid, hi);
    }
    return a[hi];
}

static int X(partition_1arg)(FLT * a, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int i = lo-1, j = hi+1;
    FLT pivot = X(selectpivot_1arg)(a, p, lo, hi, by);
    while (1) {
        do i += 1; while (by(a[i], pivot));
        do j -= 1; while (by(pivot, a[j]));
        if (i >= j) break;
        X(swap)(a, i, j);
        X(swapi)(p, i, j);
    }
    return j;
}

void X(quicksort_1arg)(FLT * a, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    if (lo < hi) {
        int mid = X(partition_1arg)(a, p, lo, hi, by);
        X(quicksort_1arg)(a, p, lo, mid, by);
        X(quicksort_1arg)(a, p, mid + 1, hi, by);
    }
}

/*
These versions of quicksort sort `a` in-place according to the `by` ordering on
`FLT` types. They also permute `b` according to the permutation required to sort
`a`, and they return the permutation `p`.
*/

static FLT X(selectpivot_2arg)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int mid = (lo+hi)/2;
    if (by(a[mid], a[lo])) {
        X(swap)(a, lo, mid);
        X(swap)(b, lo, mid);
        X(swapi)(p, lo, mid);
    }
    if (by(a[hi], a[lo])) {
        X(swap)(a, lo, hi);
        X(swap)(b, lo, hi);
        X(swapi)(p, lo, hi);
    }
    if (by(a[mid], a[hi])) {
        X(swap)(a, mid, hi);
        X(swap)(b, mid, hi);
        X(swapi)(p, mid, hi);
    }
    return a[hi];
}

static int X(partition_2arg)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int i = lo-1, j = hi+1;
    FLT pivot = X(selectpivot_2arg)(a, b, p, lo, hi, by);
    while (1) {
        do i += 1; while (by(a[i], pivot));
        do j -= 1; while (by(pivot, a[j]));
        if (i >= j) break;
        X(swap)(a, i, j);
        X(swap)(b, i, j);
        X(swapi)(p, i, j);
    }
    return j;
}

void X(quicksort_2arg)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    if (lo < hi) {
        int mid = X(partition_2arg)(a, b, p, lo, hi, by);
        X(quicksort_2arg)(a, b, p, lo, mid, by);
        X(quicksort_2arg)(a, b, p, mid + 1, hi, by);
    }
}

static FLT X(selectpivot_3arg)(FLT * a, FLT * b, FLT * c, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int mid = (lo+hi)/2;
    if (by(a[mid], a[lo])) {
        X(swap)(a, lo, mid);
        X(swap)(b, lo, mid);
        X(swap)(c, lo, mid);
        X(swapi)(p, lo, mid);
    }
    if (by(a[hi], a[lo])) {
        X(swap)(a, lo, hi);
        X(swap)(b, lo, hi);
        X(swap)(c, lo, hi);
        X(swapi)(p, lo, hi);
    }
    if (by(a[mid], a[hi])) {
        X(swap)(a, mid, hi);
        X(swap)(b, mid, hi);
        X(swap)(c, mid, hi);
        X(swapi)(p, mid, hi);
    }
    return a[hi];
}

static int X(partition_3arg)(FLT * a, FLT * b, FLT * c, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int i = lo-1, j = hi+1;
    FLT pivot = X(selectpivot_3arg)(a, b, c, p, lo, hi, by);
    while (1) {
        do i += 1; while (by(a[i], pivot));
        do j -= 1; while (by(pivot, a[j]));
        if (i >= j) break;
        X(swap)(a, i, j);
        X(swap)(b, i, j);
        X(swap)(c, i, j);
        X(swapi)(p, i, j);
    }
    return j;
}

void X(quicksort_3arg)(FLT * a, FLT * b, FLT * c, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    if (lo < hi) {
        int mid = X(partition_3arg)(a, b, c, p, lo, hi, by);
        X(quicksort_3arg)(a, b, c, p, lo, mid, by);
        X(quicksort_3arg)(a, b, c, p, mid + 1, hi, by);
    }
}

static FLT X(selectpivot_4arg)(FLT * a, FLT * b, FLT * c, FLT * d, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int mid = (lo+hi)/2;
    if (by(a[mid], a[lo])) {
        X(swap)(a, lo, mid);
        X(swap)(b, lo, mid);
        X(swap)(c, lo, mid);
        X(swap)(d, lo, mid);
        X(swapi)(p, lo, mid);
    }
    if (by(a[hi], a[lo])) {
        X(swap)(a, lo, hi);
        X(swap)(b, lo, hi);
        X(swap)(c, lo, hi);
        X(swap)(d, lo, hi);
        X(swapi)(p, lo, hi);
    }
    if (by(a[mid], a[hi])) {
        X(swap)(a, mid, hi);
        X(swap)(b, mid, hi);
        X(swap)(c, mid, hi);
        X(swap)(d, mid, hi);
        X(swapi)(p, mid, hi);
    }
    return a[hi];
}

static int X(partition_4arg)(FLT * a, FLT * b, FLT * c, FLT * d, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int i = lo-1, j = hi+1;
    FLT pivot = X(selectpivot_4arg)(a, b, c, d, p, lo, hi, by);
    while (1) {
        do i += 1; while (by(a[i], pivot));
        do j -= 1; while (by(pivot, a[j]));
        if (i >= j) break;
        X(swap)(a, i, j);
        X(swap)(b, i, j);
        X(swap)(c, i, j);
        X(swap)(d, i, j);
        X(swapi)(p, i, j);
    }
    return j;
}

void X(quicksort_4arg)(FLT * a, FLT * b, FLT * c, FLT * d, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    if (lo < hi) {
        int mid = X(partition_4arg)(a, b, c, d, p, lo, hi, by);
        X(quicksort_4arg)(a, b, c, d, p, lo, mid, by);
        X(quicksort_4arg)(a, b, c, d, p, mid + 1, hi, by);
    }
}

void X(swap)(FLT * a, int i, int j) {
    FLT temp = a[i];
    a[i] = a[j];
    a[j] = temp;
}

void X(swapi)(int * p, int i, int j) {
    int temp = p[i];
    p[i] = p[j];
    p[j] = temp;
}

int X(lt   )(FLT x, FLT y) {return x <  y;}
int X(le   )(FLT x, FLT y) {return x <= y;}
int X(gt   )(FLT x, FLT y) {return x >  y;}
int X(ge   )(FLT x, FLT y) {return x >= y;}
int X(ltabs)(FLT x, FLT y) {return Y(fabs)(x) <  Y(fabs)(y);}
int X(leabs)(FLT x, FLT y) {return Y(fabs)(x) <= Y(fabs)(y);}
int X(gtabs)(FLT x, FLT y) {return Y(fabs)(x) >  Y(fabs)(y);}
int X(geabs)(FLT x, FLT y) {return Y(fabs)(x) >= Y(fabs)(y);}
