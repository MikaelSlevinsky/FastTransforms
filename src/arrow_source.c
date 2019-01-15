void X(destroy_symmetric_arrow)(X(symmetric_arrow) * A) {
    free(A->a);
    free(A->b);
    free(A);
}

void X(destroy_upper_arrow)(X(upper_arrow) * R) {
    free(R->d);
    free(R->e);
    free(R);
}

void X(destroy_symmetric_arrow_eigen)(X(symmetric_arrow_eigen) * F) {
    free(F->Q);
    free(F->lambda);
    free(F->p);
    free(F);
}

void X(destroy_symmetric_arrow_eigen_FMM)(X(symmetric_arrow_eigen_FMM) * F) {
    X(destroy_hierarchicalmatrix)(F->Q);
    free(F->q);
    free(F->lambda);
    free(F->p);
    free(F);
}

X(upper_arrow) * X(symmetric_arrow_cholesky)(X(symmetric_arrow) * A) {
    int n = A->n;
    FLT * a = A->a;
    FLT * b = A->b;
    FLT c = A->c;
    FLT * d = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * e = (FLT *) calloc(n-1, sizeof(FLT));
    FLT f = ZERO(FLT);

    for (int i = 0; i < n-1; i++) {
        d[i] = X(sqrt)(a[i]);
        e[i] = b[i]/d[i];
        f += e[i]*e[i];
    }
    f = X(sqrt)(c - f);

    X(upper_arrow) * R = (X(upper_arrow) *) malloc(sizeof(X(upper_arrow)));
    R->d = d;
    R->e = e;
    R->f = f;
    R->n = n;
    return R;
}

X(upper_arrow) * X(upper_arrow_inv)(X(upper_arrow) * R) {
    int n = R->n;
    FLT * d = R->d;
    FLT * e = R->e;
    FLT f = R->f;
    FLT * d1 = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * e1 = (FLT *) calloc(n-1, sizeof(FLT));
    FLT f1 = ONE(FLT)/f;

    for (int i = 0; i < n-1; i++) {
        d1[i] = ONE(FLT)/d[i];
        e1[i] = -f1*d1[i]*e[i];
    }

    X(upper_arrow) * R1 = (X(upper_arrow) *) malloc(sizeof(X(upper_arrow)));
    R1->d = d1;
    R1->e = e1;
    R1->f = f1;
    R1->n = n;
    return R1;
}

X(symmetric_arrow) * X(symmetric_arrow_congruence)(X(symmetric_arrow) * A, X(symmetric_arrow) * B) {
    X(upper_arrow) * R = X(symmetric_arrow_cholesky)(B);
    X(upper_arrow) * Ri = X(upper_arrow_inv)(R);
    X(destroy_upper_arrow)(R);
    int n = A->n;
    FLT * a = A->a;
    FLT * b = A->b;
    FLT c = A->c;

    FLT * d = Ri->d;
    FLT * e = Ri->e;
    FLT f = Ri->f;

    FLT * a1 = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * b1 = (FLT *) calloc(n-1, sizeof(FLT));
    FLT c1 = c*f*f;
    FLT twof = TWO(FLT)*f;
    FLT ae = ZERO(FLT);

    for (int i = 0; i < n-1; i++) {
        a1[i] = a[i]*d[i]*d[i];
        ae = a[i]*e[i];
        b1[i] = d[i]*(ae+b[i]*f);
        c1 += e[i]*(ae+b[i]*twof);
    }

    X(destroy_upper_arrow)(Ri);

    X(symmetric_arrow) * A1 = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    A1->a = a1;
    A1->b = b1;
    A1->c = c1;
    A1->n = n;
    return A1;
}

X(symmetric_arrow) * X(symmetric_arrow_synthesize)(X(symmetric_arrow) * A, FLT * lambda) {
    int n = A->n;
    FLT * a = A->a;
    FLT * b = A->b;
    FLT c = A->c;
    FLT * as = (FLT *) calloc(n-1, sizeof(FLT));
    FLT * bs = (FLT *) calloc(n-1, sizeof(FLT));
    FLT cs = ZERO(FLT);
    FLT ai, t;

    for (int i = 0; i < n-1; i++) {
        as[i] = ai = a[i];
        cs += lambda[i] - ai;
        t = (ai-lambda[0])*(lambda[n-1]-ai);
        for (int j = 0; j < i; j++)
            t *= (lambda[j+1]-ai)/(a[j]-ai);
        for (int j = i+1; j < n-1; j++)
            t *= (lambda[j]-ai)/(a[j]-ai);
        bs[i] = X(copysign)(X(sqrt)(t), b[i]);
    }
    cs += lambda[n-1];

    X(symmetric_arrow) * AS = (X(symmetric_arrow) *) malloc(sizeof(X(symmetric_arrow)));
    AS->a = as;
    AS->b = bs;
    AS->c = cs;
    AS->n = n;
    return AS;
}

// y ← α*A*x + β*y, y ← α*Aᵀ*x + β*y
void X(samv)(char TRANS, FLT alpha, X(symmetric_arrow) * A, FLT * x, FLT beta, FLT * y) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, c = A->c;
    for (int i = 0; i < n; i++)
        y[i] = beta*y[i];
    if (TRANS == 'N' || TRANS == 'T') {
        for (int i = 0; i < n-1; i++)
            y[i] += alpha*(a[i]*x[i] + b[i]*x[n-1]);
        for (int i = 0; i < n-1; i++)
            y[n-1] += alpha*b[i]*x[i];
        y[n-1] += alpha*c*x[n-1];
    }
}

// x ← A*x, x ← Aᵀ*x
void X(uamv)(char TRANS, X(upper_arrow) * R, FLT * x) {
    int n = R->n;
    FLT * d = R->d, * e = R->e, f = R->f;
    if (TRANS == 'N') {
        for (int i = 0; i < n-1; i++)
            x[i] = d[i]*x[i] + e[i]*x[n-1];
        x[n-1] *= f;
    }
    else if (TRANS == 'T') {
        x[n-1] *= f;
        for (int i = 0; i < n-1; i++) {
            x[n-1] += e[i]*x[i];
            x[i] *= d[i];
        }
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(uasv)(char TRANS, X(upper_arrow) * R, FLT * x) {
    int n = R->n;
    FLT * d = R->d, * e = R->e, f = R->f;
    if (TRANS == 'N') {
        x[n-1] /= f;
        for (int i = 0; i < n-1; i++)
            x[i] = (x[i] - e[i]*x[n-1])/d[i];
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n-1; i++) {
            x[i] /= d[i];
            x[n-1] -= e[i]*x[i];
        }
        x[n-1] /= f;
    }
}



FLT X(secular)(X(symmetric_arrow) * A, FLT lambda) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, ret = lambda-A->c, t;
    for (int i = 0; i < n-1; i++) {
        t = b[i];
        t = t*t/(a[i]-lambda);
        ret += t;
    }
    return ret;
}

FLT * X(secular_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    FLT * ret = (FLT *) malloc(n*sizeof(FLT));
    for (int j = ib+1; j < n-1; j++)
        ret[j] = lambda[j] - A->c;
    FLT * b2 = (FLT *) malloc((n-1)*sizeof(FLT));
    for (int i = ib; i < n-1; i++)
        b2[i] = b[i]*b[i];
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(X(cauchykernel), lambda, a, (unitrange) {ib+1, n-1}, (unitrange) {ib, n-1});
    X(himv)('N', -1, H, b2+ib, 1, ret+(ib+1));
    free(b2);
    return ret;
}

FLT X(secular_derivative)(X(symmetric_arrow) * A, FLT lambda) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, ret = ONE(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = b[i]/(a[i]-lambda);
        t = t*t;
        ret += t;
    }
    return ret;
}

FLT * X(secular_derivative_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    FLT * ret = (FLT *) malloc(n*sizeof(FLT));
    for (int j = ib+1; j < n-1; j++)
        ret[j] = ONE(FLT);
    FLT * b2 = (FLT *) malloc((n-1)*sizeof(FLT));
    for (int i = ib; i < n-1; i++)
        b2[i] = b[i]*b[i];
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(X(coulombkernel), lambda, a, (unitrange) {ib+1, n-1}, (unitrange) {ib, n-1});
    X(himv)('N', 1, H, b2+ib, 1, ret+(ib+1));
    free(b2);
    return ret;
}

FLT X(secular_second_derivative)(X(symmetric_arrow) * A, FLT lambda) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, ret = ZERO(FLT), t;
    for (int i = 0; i < n-1; i++) {
        t = b[i]/(a[i]-lambda);
        t = t*t/(a[i]-lambda);
        ret += t;
    }
    return TWO(FLT)*ret;
}

FLT * X(secular_second_derivative_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    FLT * ret = (FLT *) calloc(n, sizeof(FLT));
    FLT * b2 = (FLT *) malloc((n-1)*sizeof(FLT));
    for (int i = ib; i < n-1; i++)
        b2[i] = b[i]*b[i];
    X(hierarchicalmatrix) * H = X(sample_hierarchicalmatrix)(X(coulombprimekernel), lambda, a, (unitrange) {ib+1, n-1}, (unitrange) {ib, n-1});
    X(himv)('N', -2, H, b2+ib, 1, ret+(ib+1));
    free(b2);
    return ret;
}

FLT X(first_initial_guess)(FLT a0, FLT nrmb2, FLT c) {
    FLT ret = a0;
    if (c < a0) {
        FLT amc2 = (a0-c)/2;
        ret -= amc2+X(sqrt)(amc2*amc2+nrmb2);
    }
    else {
        FLT cma2 = (c-a0)/2;
        ret -= nrmb2/(cma2+X(sqrt)(cma2*cma2+nrmb2));
    }
    return ret;
}

FLT X(first_pick_zero_update)(X(symmetric_arrow) * A, FLT lambda, int ib) {
    int n = A->n;
    FLT * Aa = A->a, * Ab = A->b;
    FLT f = X(secular)(A, lambda);
    FLT fp = X(secular_derivative)(A, lambda);
    FLT a0 = Aa[ib];
    FLT alpha = ONE(FLT), t;
    for (int i = ib+1; i < n-1; i++) {
        t = Ab[i]/(Aa[i]-lambda);
        alpha += t*t*(a0-Aa[i])/(lambda-Aa[i]);
    }
    FLT a = -alpha/(lambda-a0);
    FLT b = fp + f/(lambda-a0);
    FLT c = -f;
    return 2*c/(b+X(sqrt)(b*b-4*a*c));
}

FLT X(last_initial_guess)(FLT an, FLT nrmb2, FLT c) {
    FLT ret = an;
    if (c < an) {
        FLT amc2 = (an-c)/2;
        ret += nrmb2/(amc2+X(sqrt)(amc2*amc2+nrmb2));
    }
    else {
        FLT cma2 = (c-an)/2;
        ret += cma2+X(sqrt)(cma2*cma2+nrmb2);
    }
    return ret;
}

FLT X(last_pick_zero_update)(X(symmetric_arrow) * A, FLT lambda) {
    int n = A->n;
    FLT * Aa = A->a, * Ab = A->b;
    FLT f = X(secular)(A, lambda);
    FLT fp = X(secular_derivative)(A, lambda);
    FLT an = Aa[n-2];
    FLT alpha = ONE(FLT), t;
    for (int i = 0; i < n-2; i++) {
        t = Ab[i]/(Aa[i]-lambda);
        alpha += t*t*(an-Aa[i])/(lambda-Aa[i]);
    }
    FLT a = -alpha/(lambda-an);
    FLT b = fp + f/(lambda-an);
    FLT c = -f;
    return 2*c/(b+X(sqrt)(b*b-4*a*c));
}

FLT X(pick_zero_update)(X(symmetric_arrow) * A, FLT lambda, int j) {
    FLT * Aa = A->a;
    FLT f = X(secular)(A, lambda);
    FLT fp = X(secular_derivative)(A, lambda);
    FLT fpp = X(secular_second_derivative)(A, lambda);
    FLT c1 = 1/(Aa[j-1] - lambda);
    FLT c2 = 1/(Aa[j] - lambda);
    FLT c2g = (fpp - 2*c1*fp)/(2*(c2-c1)*c2);
    FLT c1b = (fp-c2*c2g)/c1;
    FLT alpha = f - c1b - c2g;
    FLT a = alpha/((Aa[j-1] - lambda)*(lambda - Aa[j]));
    FLT b = fp - (c1+c2)*f;
    FLT c = -f;
    return 2*c/(b+X(sqrt)(b*b-4*a*c));
}

FLT * X(pick_zero_update_FMM)(X(symmetric_arrow) * A, FLT * lambda, int ib) {
    int n = A->n;
    FLT * Aa = A->a;
    FLT * f = X(secular_FMM)(A, lambda, ib);
    FLT * fp = X(secular_derivative_FMM)(A, lambda, ib);
    FLT * fpp = X(secular_second_derivative_FMM)(A, lambda, ib);
    FLT c1, c2, c2g, c1b, alpha, a, b, c;
    FLT * ret = (FLT *) malloc((n-1)*sizeof(FLT));
    for (int j = ib+1; j < n-1; j++) {
        c1 = 1/(Aa[j-1] - lambda[j]);
        c2 = 1/(Aa[j] - lambda[j]);
        c2g = (fpp[j] - 2*c1*fp[j])/(2*(c2-c1)*c2);
        c1b = (fp[j]-c2*c2g)/c1;
        alpha = f[j] - c1b - c2g;
        a = alpha/((Aa[j-1] - lambda[j])*(lambda[j] - Aa[j]));
        b = fp[j] - (c1+c2)*f[j];
        c = -f[j];
        ret[j] = 2*c/(b+X(sqrt)(b*b-4*a*c));
    }
    free(f);
    free(fp);
    free(fpp);
    return ret;
}


// Note: This modifies `A`.
int X(symmetric_arrow_deflate)(X(symmetric_arrow) * A, int * p) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, c = A->c;
    FLT nrmb = 0;
    int ib;
    for (int i = 0; i < n-1; i++)
        nrmb += b[i]*b[i];
    nrmb = X(sqrt)(nrmb);

    // absolute sort based on the spike `b`.
    X(quicksort)(b, a, p, 0, n-2, X(ltabs));

    // `|b[ib]|` is the first value that surpasses `||b||_2 × ϵ` in magnitude.
    for (ib = 0; ib < n-1; ib++)
        if (X(fabs)(b[ib]) > nrmb*X(eps)()) break;

    // re-sort based on ensuring the elements of `a` with non-negligible
    // corresponding `b` are increasing.
    X(quicksort)(a, b, p, ib, n-2, X(lt));

    return ib;
}

// Assuming that ib = X(symmetric_arrow_deflate)(A, p); has been called.
FLT * X(symmetric_arrow_eigvals)(X(symmetric_arrow) * A, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, c = A->c;
    int n2 = n*n;

    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));

    FLT nrmb2 = 0;
    for (int i = 0; i < n-1; i++)
        nrmb2 += b[i]*b[i];

    for (int i = 0; i < ib; i++)
        lambda[i] = a[i];

    FLT lambdak, deltak;

    lambdak = X(first_initial_guess)(a[ib], nrmb2, c);
    deltak = ONE(FLT);
    while (X(fabs)(deltak) > n2*X(eps)()) {
        deltak = X(first_pick_zero_update)(A, lambdak, ib);
        if (deltak != deltak) break;
        lambdak += deltak;
    }
    lambda[ib] = lambdak;

    #pragma omp parallel for
    for (int j = ib+1; j < n-1; j++) {
        FLT lambdak = (a[j]+a[j-1])/2;
        FLT deltak = ONE(FLT);
        while (X(fabs)(deltak) > n2*X(eps)()) {
            deltak = X(pick_zero_update)(A, lambdak, j);
            if (deltak != deltak) break;
            lambdak += deltak;
        }
        lambda[j] = lambdak;
    }

    lambdak = X(last_initial_guess)(a[n-2], nrmb2, c);
    deltak = ONE(FLT);
    while (X(fabs)(deltak) > n2*X(eps)()) {
        deltak = X(last_pick_zero_update)(A, lambdak);
        if (deltak != deltak) break;
        lambdak += deltak;
    }
    lambda[n-1] = lambdak;

    return lambda;
}

// Assuming that ib = X(symmetric_arrow_deflate)(A, p); has been called.
FLT * X(symmetric_arrow_eigvals_FMM)(X(symmetric_arrow) * A, int ib) {
    int n = A->n;
    FLT * a = A->a, * b = A->b, c = A->c;
    int n2 = n*n;

    FLT * lambda = (FLT *) calloc(n, sizeof(FLT));

    FLT nrmb2 = 0;
    for (int i = 0; i < n-1; i++)
        nrmb2 += b[i]*b[i];

    for (int i = 0; i < ib; i++)
        lambda[i] = a[i];

    FLT lambdak, deltak;

    lambdak = X(first_initial_guess)(a[ib], nrmb2, c);
    deltak = ONE(FLT);
    while (X(fabs)(deltak) > 100*n2*X(eps)()) {
        deltak = X(first_pick_zero_update)(A, lambdak, ib);
        if (deltak != deltak) break;
        lambdak += deltak;
    }
    lambda[ib] = lambdak;

    FLT nrmd = 1;
    for (int j = ib+1; j < n-1; j++)
        lambda[j] = (a[j]+a[j-1])/2;
    while (nrmd > 100*n2*X(eps)()) {
        FLT * delta = X(pick_zero_update_FMM)(A, lambda, ib);
        nrmd = 0;
        for (int j = ib+1; j < n-1; j++) {
            if (delta[j] != delta[j]) continue;
            lambda[j] += delta[j];
            nrmd += delta[j]*delta[j];
        }
        nrmd = X(sqrt)(nrmd);
        free(delta);
    }

    lambdak = X(last_initial_guess)(a[n-2], nrmb2, c);
    deltak = ONE(FLT);
    while (X(fabs)(deltak) > 100*n2*X(eps)()) {
        deltak = X(last_pick_zero_update)(A, lambdak);
        if (deltak != deltak) break;
        lambdak += deltak;
    }
    lambda[n-1] = lambdak;

    return lambda;
}

FLT * X(symmetric_arrow_eigvecs)(X(symmetric_arrow) * A, FLT * lambda, int ib) {
    int n = A->n;//, k;
    FLT * a = A->a, * b = A->b, c = A->c;
    FLT * Q = (FLT *) calloc(n*n, sizeof(FLT));
    FLT nrm;
    for (int j = 0; j < ib; j++)
        Q[j+j*n] = 1;
    for (int j = ib; j < n; j++) {
        for (int i = ib; i < n-1; i++)
            Q[i+j*n] = b[i]/(lambda[j]-a[i]);
        Q[n-1+j*n] = 1;
        nrm = 0;
        for (int i = ib; i < n; i++)
            nrm += Q[i+j*n]*Q[i+j*n];
        nrm = 1/X(sqrt)(nrm);
        for (int i = ib; i < n; i++)
            Q[i+j*n] *= nrm;
        /*
        k = -1;
        for (int i = 0; i < n; i++) {
            if (X(isfinite)(Q[i+j*n])) nrm += Q[i+j*n]*Q[i+j*n];
            else {k = i; break;}
        }
        if (k != -1) {
            for (int i = 0; i < n; i++)
                Q[i+j*n] = 0;
            Q[k+j*n] = 1;
        }
        else {
            nrm = 1/X(sqrt)(nrm);
            for (int i = 0; i < n; i++)
                Q[i+j*n] *= nrm;
        }
        */
    }
    return Q;
}

// Note: this modifies `A`.
X(symmetric_arrow_eigen) * X(symmetric_arrow_eig)(X(symmetric_arrow) * A) {
    int n = A->n;
    int * q = (int *) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        q[i] = i;
    int ib = X(symmetric_arrow_deflate)(A, q);
    FLT * lambda = X(symmetric_arrow_eigvals)(A, ib);
    X(symmetric_arrow) * B = X(symmetric_arrow_synthesize)(A, lambda);
    int * p = (int *) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        p[i] = i;
    ib = X(symmetric_arrow_deflate)(B, p);
    for (int i = 0; i < n; i++)
        p[i] = q[p[i]];
    free(q);
    free(lambda);
    lambda = X(symmetric_arrow_eigvals)(B, ib);
    FLT * Q = X(symmetric_arrow_eigvecs)(B, lambda, ib);
    X(destroy_symmetric_arrow)(B);
    X(symmetric_arrow_eigen) * F = (X(symmetric_arrow_eigen) *) malloc(sizeof(X(symmetric_arrow_eigen)));
    F->Q = Q;
    F->lambda = lambda;
    F->p = p;
    F->n = n;
    return F;
}

X(symmetric_arrow_eigen_FMM) * X(symmetric_arrow_eig_FMM)(X(symmetric_arrow) * A) {
    int n = A->n;
    int * pq = (int *) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        pq[i] = i;
    int ib = X(symmetric_arrow_deflate)(A, pq);
    FLT * lambda = X(symmetric_arrow_eigvals_FMM)(A, ib);
    X(symmetric_arrow) * B = X(symmetric_arrow_synthesize)(A, lambda);
    int * p = (int *) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        p[i] = i;
    ib = X(symmetric_arrow_deflate)(B, p);
    for (int i = 0; i < n; i++)
        p[i] = pq[p[i]];
    free(pq);
    free(lambda);
    lambda = X(symmetric_arrow_eigvals_FMM)(B, ib);

    FLT * a = B->a, * b = B->b;
    X(hierarchicalmatrix) * Q = X(sample_hierarchicalmatrix)(X(cauchykernel), a, lambda, (unitrange) {ib, n-1}, (unitrange) {ib, n});
    X(hierarchicalmatrix) * N = X(sample_hierarchicalmatrix)(X(coulombkernel), lambda, a, (unitrange) {ib, n}, (unitrange) {ib, n-1});
    FLT * b2 = (FLT *) calloc(n-1-ib, sizeof(FLT));
    for (int i = ib; i < n-1; i++)
        b2[i-ib] = b[i]*b[i];
    FLT * q = (FLT *) calloc(n-ib, sizeof(FLT));
    X(himv)('N', 1, N, b2, 0, q);
    free(b2);
    for (int i = ib; i < n; i++)
        q[i-ib] = 1/X(sqrt)(1+q[i-ib]);
    X(scale_rows_hierarchicalmatrix)(-1, b+ib, Q);
    X(scale_columns_hierarchicalmatrix)(1, q, Q);

    X(destroy_symmetric_arrow)(B);
    X(destroy_hierarchicalmatrix)(N);

    X(symmetric_arrow_eigen_FMM) * F = (X(symmetric_arrow_eigen_FMM) *) malloc(sizeof(X(symmetric_arrow_eigen_FMM)));
    F->Q = Q;
    F->q = q;
    F->lambda = lambda;
    F->p = p;
    F->n = n;
    F->ib = ib;
    return F;
}

/*
These versions of quicksort sort `a` in-place according to the `by` ordering on
`FLT` types. They also permute `b` according to the permutation required to sort
`a`, and they return the permutation `p`.
*/

void X(quicksort)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    if (lo < hi) {
        int mid = X(partition)(a, b, p, lo, hi, by);
        X(quicksort)(a, b, p, lo, mid, by);
        X(quicksort)(a, b, p, mid + 1, hi, by);
    }
}

int X(partition)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int i = lo-1, j = hi+1;
    FLT pivot = X(selectpivot)(a, b, p, lo, hi, by);
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

FLT X(selectpivot)(FLT * a, FLT * b, int * p, int lo, int hi, int (*by)(FLT x, FLT y)) {
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
int X(ltabs)(FLT x, FLT y) {return X(fabs)(x) <  X(fabs)(y);}
int X(leabs)(FLT x, FLT y) {return X(fabs)(x) <= X(fabs)(y);}
int X(gtabs)(FLT x, FLT y) {return X(fabs)(x) >  X(fabs)(y);}
int X(geabs)(FLT x, FLT y) {return X(fabs)(x) >= X(fabs)(y);}
