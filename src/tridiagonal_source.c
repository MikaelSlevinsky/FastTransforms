void X(destroy_symmetric_tridiagonal)(X(symmetric_tridiagonal) * A) {
    free(A->a);
    free(A->b);
    free(A);
}

void X(destroy_bidiagonal)(X(bidiagonal) * B) {
    free(B->c);
    free(B->d);
    free(B);
}

void X(destroy_symmetric_tridiagonal_symmetric_eigen)(X(symmetric_tridiagonal_symmetric_eigen) * F) {
    free(F->A);
    free(F->B);
    free(F->C);
    free(F->lambda);
    free(F->phi0);
    free(F);
}

X(bidiagonal) * X(symmetric_tridiagonal_cholesky)(X(symmetric_tridiagonal) * A) {
    int n = A->n;
    FLT * a = A->a;
    FLT * b = A->b;
    FLT * c = malloc(n*sizeof(FLT));
    FLT * d = malloc((n-1)*sizeof(FLT));

    c[0] = Y(sqrt)(a[0]);
    for (int i = 0; i < n-1; i++) {
        d[i] = b[i]/c[i];
        c[i+1] = Y(sqrt)(a[i+1]-d[i]*d[i]);
    }

    X(bidiagonal) * B = malloc(sizeof(X(bidiagonal)));
    B->c = c;
    B->d = d;
    B->n = n;
    return B;
}

// y ← α*A*x + β*y, y ← α*Aᵀ*x + β*y
void X(stmv)(char TRANS, FLT alpha, X(symmetric_tridiagonal) * A, FLT * x, FLT beta, FLT * y) {
    int n = A->n;
    FLT * a = A->a, * b = A->b;
    for (int i = 0; i < n; i++)
        y[i] = beta*y[i];
    if (TRANS == 'N' || TRANS == 'T') {
        y[0] += alpha*(a[0]*x[0] + b[0]*x[1]);
        for (int i = 1; i < n-1; i++)
            y[i] += alpha*(b[i-1]*x[i-1] + a[i]*x[i] + b[i]*x[i+1]);
        y[n-1] += alpha*(b[n-2]*x[n-2] + a[n-1]*x[n-1]);
    }
}

// x ← A*x, x ← Aᵀ*x
void X(bdmv)(char TRANS, X(bidiagonal) * B, FLT * x) {
    int n = B->n;
    FLT * c = B->c, * d = B->d;
    if (TRANS == 'N') {
        for (int i = 0; i < n-1; i++)
            x[i] = c[i]*x[i] + d[i]*x[i+1];
        x[n-1] *= c[n-1];
    }
    else if (TRANS == 'T') {
        for (int i = n-1; i > 0; i--)
            x[i] = d[i-1]*x[i-1] + c[i]*x[i];
        x[0] *= c[0];
    }
}

// x ← A⁻¹*x, x ← A⁻ᵀ*x
void X(bdsv)(char TRANS, X(bidiagonal) * B, FLT * x) {
    int n = B->n;
    FLT * c = B->c, * d = B->d;
    if (TRANS == 'N') {
        x[n-1] /= c[n-1];
        for (int i = n-2; i >= 0; i--)
            x[i] = (x[i] - d[i]*x[i+1])/c[i];
    }
    else if (TRANS == 'T') {
        x[0] /= c[0];
        for (int i = 1; i < n; i++)
            x[i] = (x[i] - d[i-1]*x[i-1])/c[i];
    }
}

/*
`symmetric_tridiagonal_eig` is an adaptation of EISPACK's symmetric tridiagonal
QL algorithm to function with any floating-point type that may be placed in by
a C pre-processor.

According to J. Dongarra, EISPACK is distributed under the
Modified BSD or MIT license:

http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg01379.html

though the e-mail exchange does not reveal a literal license document. On the
other hand, EISPACK is distributed without a license according to C. Moler:

https://blogs.mathworks.com/cleve/2018/01/02/eispack-matrix-eigensystem-routines/

The Fortran source code is here:

http://www.netlib.org/cgi-bin/netlibfiles.pl?filename=/eispack/tql2.f
*/

#define V(i, j) V[(i)+(n)*(j)]

void X(symmetric_tridiagonal_eig)(X(symmetric_tridiagonal) * A, FLT * V, FLT * lambda) {
    int n = A->n;
    FLT * d = malloc(n*sizeof(FLT));
    FLT * e = malloc(n*sizeof(FLT));

    int i,j,k,l,m,ii,l1,l2,mml;
    FLT c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2;

    for (int i = 0; i < n; i++)
        d[i] = A->a[i];
    for (int i = 0; i < n-1; i++)
        e[i] = A->b[i];
    e[n-1] = ZERO(FLT);
    f = ZERO(FLT);
    tst1 = ZERO(FLT);

    if (n == 1) goto L1000;

    for (l = 0; l < n; l++) {
        j = 0;
        h = Y(fabs)(d[l]) + Y(fabs)(e[l]);
        if (tst1 < h) tst1 = h;
        for (m = l; m < n; m++) {
            tst2 = tst1 + Y(fabs)(e[m]);
            if (tst2 == tst1) goto L120;
        }
        L120: if (m == l) goto L220;
        L130: if (j == 60) goto L1000;
        j = j + 1;
        l1 = l + 1;
        l2 = l1 + 1;
        g = d[l];
        p = (d[l1] - g)/ (TWO(FLT)*e[l]);
        r = Y(hypot)(p, ONE(FLT));
        d[l] = e[l] / (p + Y(copysign)(r, p));
        d[l1] = e[l] * (p + Y(copysign)(r, p));
        dl1 = d[l1];
        h = g - d[l];
        for (i = l2; i < n; i++)
            d[i] = d[i] - h;
        f = f + h;
        p = d[m];
        c = ONE(FLT);
        c2 = c;
        el1 = e[l1];
        s = ZERO(FLT);
        mml = m - l;
        for (ii = 1; ii <= mml; ii++) {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c*e[i];
            h = c*p;
            r = Y(hypot)(p, e[i]);
            e[i+1] = s*r;
            s = e[i]/r;
            c = p/r;
            p = c*d[i] - s*g;
            d[i+1] = h + s*(c*g + s*d[i]);
            for (k = 0; k < n; k++) {
                h = V(k,i+1);
                V(k,i+1) = s*V(k,i) + c*h;
                V(k,i) = c*V(k,i) - s*h;
            }
        }
        p = -s*s2*c3*el1*e[l]/dl1;
        e[l] = s*p;
        d[l] = c*p;
        tst2 = tst1 + Y(fabs)(e[l]);
        if (tst2 > tst1) goto L130;
        L220: d[l] = d[l] + f;
    }
    for (ii = 1; ii < n; ii++) {
        i = ii - 1;
        k = i;
        p = d[i];
        for (j = ii; j < n; j++) {
            if (d[j] >= p) goto L260;
            k = j;
            p = d[j];
        L260: ;}
        if (k == i) goto L300;
        d[k] = d[i];
        d[i] = p;
        for (j = 0; j < n; j++) {
            p = V(j,i);
            V(j,i) = V(j,k);
            V(j,k) = p;
        }
    L300: ;}
    L1000: ;
    for (i = 0; i < n; i++)
        lambda[i] = d[i];
    free(d);
    free(e);
    for (j = 0; j < n; j++)
        if (Y(signbit)(V(j,j)))
            for (i = 0; i < n; i++)
                V(i,j) = -V(i,j);
}

void X(symmetric_definite_tridiagonal_eig)(X(symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B, FLT * V, FLT * lambda) {
    X(symmetric_tridiagonal) * AS = X(symmetric_tridiagonal_congruence)(A, B, V);
    X(symmetric_tridiagonal_eig)(AS, V, lambda);
    X(destroy_symmetric_tridiagonal)(AS);
}

X(symmetric_tridiagonal) * X(symmetric_tridiagonal_congruence)(X(symmetric_tridiagonal) * A, X(symmetric_tridiagonal) * B, FLT * V) {
    X(bidiagonal) * R = X(symmetric_tridiagonal_cholesky)(B);
    int n = A->n;
    FLT * c = R->c;
    FLT * d = R->d;
    FLT * a = malloc(n*sizeof(FLT));
    FLT * b = malloc((n-1)*sizeof(FLT));
    FLT co, si, r, t1, t2, t3, t4, w;
    for (int i = 0; i < n-1; i++) {
        a[i] = A->a[i];
        b[i] = A->b[i];
    }
    a[n-1] = A->a[n-1];

    // Step 1: first elementary Cholesky factor. Introduces no fill-in.
    a[0] = a[0]/(c[0]*c[0]);
    t1 = b[0]/c[0];
    t2 = a[0]*d[0];
    b[0] = t1-t2;
    a[1] = a[1] + d[0]*(t2-TWO(FLT)*t1);
    // Build the elementary factor into generalized eigenvectors.
    for (int k = 0; k < n; k++) {
        V(k,0) = V(k,0)/c[0];
        V(k,1) = V(k,1) - d[0]*V(k,0);
    }
    // Step 2: loop through all remaining (but last) elementary Cholesky factors. Introduce a small bulge that is chased upward via Givens rotations.
    for (int i = 1; i < n-1; i++) {
        b[i-1] = b[i-1]/c[i];
        w = -b[i-1]*d[i];
        a[i] = a[i]/(c[i]*c[i]);
        t1 = b[i]/c[i];
        t2 = a[i]*d[i];
        b[i] = t1 - t2;
        a[i+1] = a[i+1] + d[i]*(t2-TWO(FLT)*t1);
        // Build the elementary factor into generalized eigenvectors.
        for (int k = 0; k < n; k++) {
            V(k,i) = V(k,i)/c[i];
            V(k,i+1) = V(k,i+1) - d[i]*V(k,i);
        }
        for (int j = i; j > 1; j--) {
            // Apply Givens.
            r = Y(hypot)(b[j], w);
            if (r != ZERO(FLT)) {
                co = b[j]/r;
                si = -w/r;
                t1 = co*a[j-1]+si*b[j-1];
                t2 = co*b[j-1]+si*a[j  ];
                t3 = co*a[j  ]-si*b[j-1];
                t4 = co*b[j-1]-si*a[j-1];
                a[j  ] = co*t3-si*t4;
                a[j-1] = co*t1+si*t2;
                b[j-1] = co*t2-si*t1;
                b[j  ] = r;
                w = -si*b[j-2];
                b[j-2] = co*b[j-2];
                for (int k = 0; k < n; k++) {
                    t1 = V(k,j-1);
                    V(k,j-1) = co*t1 + si*V(k,j);
                    V(k,j  ) = co*V(k,j) - si*t1;
                }
            }
        }
        // Special rotation at the end. Bulge completely chased up; does not access b[-1].
        r = Y(hypot)(b[1], w);
        if (r != ZERO(FLT)) {
            co = b[1]/r;
            si = -w/r;
            t1 = co*a[0]+si*b[0];
            t2 = co*b[0]+si*a[1];
            t3 = co*a[1]-si*b[0];
            t4 = co*b[0]-si*a[0];
            a[1] = co*t3-si*t4;
            a[0] = co*t1+si*t2;
            b[0] = co*t2-si*t1;
            b[1] = r;
            for (int k = 0; k < n; k++) {
                t1 = V(k,0);
                V(k,0) = co*t1 + si*V(k,1);
                V(k,1) = co*V(k,1) - si*t1;
            }
        }
    }
    // Step 3: last elementary Cholesky factor. Introduces no fill-in.
    b[n-2] = b[n-2]/c[n-1];
    a[n-1] = a[n-1]/(c[n-1]*c[n-1]);
    for (int k = 0; k < n; k++)
        V(k,n-1) = V(k,n-1)/c[n-1];

    X(destroy_bidiagonal)(R);

    X(symmetric_tridiagonal) * A1 = malloc(sizeof(X(symmetric_tridiagonal)));
    A1->a = a;
    A1->b = b;
    A1->n = n;
    return A1;
}

#undef V

X(symmetric_tridiagonal_symmetric_eigen) * X(symmetric_tridiagonal_symmetric_eig)(X(symmetric_tridiagonal) * T, FLT * lambda, const int sign) {
    int n = T->n;
    FLT * A = calloc(n, sizeof(FLT));
    FLT * B = calloc(n, sizeof(FLT));
    FLT * C = calloc(n+1, sizeof(FLT));
    A[0] = 1/T->b[0];
    B[0] = -T->a[0]/T->b[0];
    for (int i = 1; i < n-1; i++) {
        A[i] = 1/T->b[i];
        B[i] = -T->a[i]/T->b[i];
        C[i] = T->b[i-1]/T->b[i];
    }
    FLT * phi0 = malloc(n*sizeof(FLT));
    FLT nrm = phi0[0] = 1;
    if (n > 1) {
        phi0[1] = A[0]*lambda[0]+B[0];
        nrm += phi0[1]*phi0[1];
    }
    for (int i = 1; i < n-1; i++) {
        phi0[i+1] = (A[i]*lambda[0]+B[i])*phi0[i] - C[i]*phi0[i-1];
        nrm += phi0[i+1]*phi0[i+1];
    }
    nrm = (sign > 0) ? 1/Y(sqrt)(nrm) : -1/Y(sqrt)(nrm);
    for (int i = 0; i < n; i++)
        phi0[i] *= nrm;
    X(symmetric_tridiagonal_symmetric_eigen) * F = malloc(sizeof(X(symmetric_tridiagonal_symmetric_eigen)));
    F->A = A;
    F->B = B;
    F->C = C;
    F->lambda = malloc(n*sizeof(FLT));
    for (int i = 0; i < n; i++)
        F->lambda[i] = lambda[i];
    F->phi0 = phi0;
    F->n = n;
    return F;
}

// y = V*x == Vᵀx
void X(semv)(X(symmetric_tridiagonal_symmetric_eigen) * F, FLT * x, int incx, FLT * y) {
    X(orthogonal_polynomial_clenshaw)(F->n, x, incx, F->A, F->B, F->C, F->n, F->lambda, F->phi0, y);
}

X(symmetric_tridiagonal) * X(create_A_shtsdtev)(const int n, const int mu, const int m, char PARITY) {
    X(symmetric_tridiagonal) * A = malloc(sizeof(X(symmetric_tridiagonal)));
    int shft;
    FLT * a = calloc(n, sizeof(FLT));
    FLT * b = calloc(n-1, sizeof(FLT));
    FLT rat, num, den, ld, md = (FLT) m;

    if (PARITY == 'E') shft = 0;
    else if (PARITY == 'O') shft = 1;

    for (int l = 1+shft; l < 2*n+1+shft; l += 2) {
        ld = l;
        num = 2*ld*(ld+1)*(ld*ld+ld-1) + md*(8*ld*ld*ld+8*ld*ld-4*ld + md*(14*ld*ld+6*ld-6 + md*(12*ld+2 + 4*md)));
        den = (2*ld+2*md-1)*(2*ld+2*md+3);
        a[(l-1)/2] = num/den + (mu-md)*(mu+md);
    }
    for (int l = 1+shft; l < 2*n-1+shft; l += 2) {
        ld = l;
        rat = (ld/(2*ld+2*md+1))*((ld+1)/(2*ld+2*md+3))*((ld+2*md+2)/(2*ld+2*md+3))*((ld+2*md+3)/(2*ld+2*md+5));
        b[(l-1)/2] = -(ld+md+1)*(ld+md+2)*Y(sqrt)(rat);
    }

    A->n = n;
    A->a = a;
    A->b = b;
    return A;
}

X(symmetric_tridiagonal) * X(create_B_shtsdtev)(const int n, const int m, char PARITY) {
    X(symmetric_tridiagonal) * B = malloc(sizeof(X(symmetric_tridiagonal)));
    int shft;
    FLT * a = calloc(n, sizeof(FLT));
    FLT * b = calloc(n-1, sizeof(FLT));
    FLT rat, num, den, ld, md = (FLT) m;

    if (PARITY == 'E') shft = 0;
    else if (PARITY == 'O') shft = 1;

    for (int l = 1+shft; l < 2*n+1+shft; l += 2) {
        ld = l;
        num = 2*(ld*(ld+1) + md*(2*ld+3 + 2*md));
        den = (2*ld+2*md-1)*(2*ld+2*md+3);
        a[(l-1)/2] = num/den;
    }
    for (int l = 1+shft; l < 2*n-1+shft; l += 2) {
        ld = l;
        rat = (ld/(2*ld+2*md+1))*((ld+1)/(2*ld+2*md+3))*((ld+2*md+2)/(2*ld+2*md+3))*((ld+2*md+3)/(2*ld+2*md+5));
        b[(l-1)/2] = -Y(sqrt)(rat);
    }

    B->n = n;
    B->a = a;
    B->b = b;
    return B;
}

X(bidiagonal) * X(create_R_shtsdtev)(const int n, const int m, char PARITY) {
    X(bidiagonal) * R = malloc(sizeof(X(bidiagonal)));
    int shft;
    FLT * c = calloc(n, sizeof(FLT));
    FLT * d = calloc(n-1, sizeof(FLT));
    FLT rat, ld, md = (FLT) m;

    if (PARITY == 'E') shft = 0;
    else if (PARITY == 'O') shft = 1;

    for (int l = 1+shft; l < 2*n+1+shft; l += 2) {
        ld = l;
        rat = ((ld+2*md)/(2*ld+2*md-1))*((ld+2*md+1)/(2*ld+2*md+1));
        c[(l-1)/2] = Y(sqrt)(rat);
    }
    for (int l = 1+shft; l < 2*n-1+shft; l += 2) {
        ld = l;
        rat = (ld/(2*ld+2*md+1))*((ld+1)/(2*ld+2*md+3));
        d[(l-1)/2] = -Y(sqrt)(rat);
    }

    R->n = n;
    R->c = c;
    R->d = d;
    return R;
}
