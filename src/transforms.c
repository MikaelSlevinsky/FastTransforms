// Computational routines for one-dimensional orthogonal polynomial transforms.

#include "fasttransforms.h"
#include "ftinternal.h"

static inline double stirlingseries(const double z) {
    double iz = 1.0/z;
    if (z >= 3274.12075200175)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273))));
    else if (z >= 590.1021805526798)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917)))));
    else if (z >= 195.81733962412835)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666))))));
    else if (z >= 91.4692823071966)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5)))))));
    else if (z >= 52.70218954633605)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939))))))));
    else if (z >= 34.84031591198865)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5)))))))));
    else if (z >= 25.3173982783047)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873))))))))));
    else if (z >= 19.685015283078513)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5)))))))))));
    else if (z >= 16.088669099569266)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5 + iz*(-0.0019144384985654776))))))))))));
    else if (z >= 13.655055978888104)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5 + iz*(-0.0019144384985654776 + iz*(-0.00016251626278391583)))))))))))));
    else if (z >= 11.93238782087875)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5 + iz*(-0.0019144384985654776 + iz*(-0.00016251626278391583 + iz*(0.00640336283380807))))))))))))));
    else if (z >= 10.668852439197263)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5 + iz*(-0.0019144384985654776 + iz*(-0.00016251626278391583 + iz*(0.00640336283380807 + iz*(0.0005401647678926045)))))))))))))));
    else if (z >= 9.715358216638403)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5 + iz*(-0.0019144384985654776 + iz*(-0.00016251626278391583 + iz*(0.00640336283380807 + iz*(0.0005401647678926045 + iz*(-0.02952788094569912))))))))))))))));
    else if (z >= 8.979120323411497)
        return (1.0 + iz*(0.08333333333333333 + iz*(0.003472222222222222 + iz*(-0.0026813271604938273 + iz*(-0.00022947209362139917 + iz*(0.0007840392217200666 + iz*(6.972813758365857e-5 + iz*(-0.0005921664373536939 + iz*(-5.171790908260592e-5 + iz*(0.0008394987206720873 + iz*(7.204895416020011e-5 + iz*(-0.0019144384985654776 + iz*(-0.00016251626278391583 + iz*(0.00640336283380807 + iz*(0.0005401647678926045 + iz*(-0.02952788094569912 + iz*(-0.002481743600264998)))))))))))))))));
    else
        return 0.0;
}

static inline double Aratio(const int n, const double alpha, const double beta) {
    return exp((0.5*n + alpha + 0.25)*log1p(-beta/(n + alpha + beta + 1.0)) + (0.5*n + beta + 0.25)*log1p(-alpha/(n + alpha + beta + 1.0)) + (0.5*n + 0.25)*log1p(alpha/(n + 1.0))+(0.5*n + 0.25)*log1p(beta/(n + 1.0)));
}

static inline double Analphabeta(const int n, const double alpha, const double beta) {
    if (n == 0 && alpha + beta == -1.0)
        return tgamma(alpha + 1.0)*tgamma(beta + 1.0);
    double t = fmin(fmin(fmin(alpha, beta), alpha + beta), 0.0);
    if (n + t >= 7.979120323411497)
        return pow(2.0, alpha + beta + 1.0)/(2.0*n + alpha + beta + 1.0)*stirlingseries(n + alpha + 1.0)*Aratio(n, alpha, beta)/stirlingseries(n + alpha + beta + 1.0)*stirlingseries(n + beta + 1.0)/stirlingseries(n+1.0);
    return (n + 1.0)*(n + alpha + beta + 1.0)/(n + alpha + 1.0)/(n + beta + 1.0)*Analphabeta(n+1, alpha, beta)*((2.0*n + alpha + beta + 3.0)/(2.0*n + alpha + beta + 1.0));
}

static inline double lambda(const double x) {
    if (x > 9.84475) {
        double xp = x + 0.25;
        double invxp2 = 1.0/(xp*xp);
        return (1.0 + invxp2*(-1.5625e-02 + invxp2*(2.5634765625e-03 + invxp2*(-1.2798309326171875e-03 + invxp2*(1.343511044979095458984375e-03 + invxp2*(-2.432896639220416545867919921875e-03 + invxp2*6.7542375336415716446936130523681640625e-03))))))/sqrt(xp);
    }
    return (x + 1.0)*lambda(x + 1.0)/(x + 0.5);
}

static inline double lambda2(const double x, const double l1, const double l2) {
    if (fmin(x + l1, x + l2) >= 8.979120323411497)
        return exp(l2 - l1 + (x - 0.5)*log1p((l1 - l2)/(x + l2)))*pow(x + l1, l1)/pow(x + l2, l2)*stirlingseries(x + l1)/stirlingseries(x + l2);
    return (x + l2)/(x + l1)*lambda2(x + 1.0, l1, l2);
}

#define A(i,j) A[(i)+n*(j)]

double * plan_leg2cheb(const int normleg, const int normcheb, const int n) {
    double * A = (double *) calloc(n * n, sizeof(double));
    double * lam = (double *) calloc(n, sizeof(double));
    double * sclrow = (double *) calloc(n, sizeof(double));
    double * sclcol = (double *) calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        lam[i] = lambda((double) i);
        sclrow[i] = normcheb ? i ? M_SQRT_PI/M_SQRT2 : M_SQRT_PI : 1.0;
        sclcol[i] = normleg ? sqrt(i+0.5) : 1.0;
    }
    for (int j = 0; j < n; j++) {
        for (int i = j; i >= 0; i -= 2)
            A(i,j) = sclrow[i]*lam[(j-i)/2]*lam[(j+i)/2]*M_2_PI*sclcol[j];
        A(0,j) *= 0.5;
    }
    free(lam);
    free(sclrow);
    free(sclcol);
    return A;
}

double * plan_cheb2leg(const int normcheb, const int normleg, const int n) {
    double * A = (double *) calloc(n * n, sizeof(double));
    double * lam = (double *) calloc(n, sizeof(double));
    double * lamh = (double *) calloc(n, sizeof(double));
    double * sclrow = (double *) calloc(n, sizeof(double));
    double * sclcol = (double *) calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        lam[i] = lambda((double) i);
        lamh[i] = lambda(i - 0.5);
        sclrow[i] = normleg ? 1.0/sqrt(i+0.5) : 1.0;
        sclcol[i] = normcheb ? i ? M_SQRT2/M_SQRT_PI : M_1_SQRT_PI : 1.0;
    }
    if (n > 0)
        A(0,0) = sclrow[0]*1.0*sclcol[0];
    if (n > 1)
        A(1,1) = sclrow[1]*1.0*sclcol[1];
    for (int j = 2; j < n; j++)
        A(j,j) = sclrow[j]*M_SQRT_PI_2/lam[j]*sclcol[j];
    for (int j = 2; j < n; j++)
        for (int i = j-2; i >= 0; i -= 2)
            A(i,j) = -sclrow[i]*(i+0.5)*(lam[(j-i-2)/2]/(j-i))*(lamh[(j+i)/2]/(j+i+1))*j*sclcol[j];
    free(lam);
    free(lamh);
    free(sclrow);
    free(sclcol);
    return A;
}

double * plan_ultra2ultra(const int normultra1, const int normultra2, const int n, const double l1, const double l2) {
    double * A = (double *) calloc(n * n, sizeof(double));
    double * lam1 = (double *) calloc(n, sizeof(double));
    double * lam2 = (double *) calloc(n, sizeof(double));
    double * sclrow = (double *) calloc(n, sizeof(double));
    double * sclcol = (double *) calloc(n, sizeof(double));
    double scl = tgamma(l2)/tgamma(l1);
    for (int i = 0; i < n; i++) {
        lam1[i] = lambda2((double) i, l1 - l2, 1.0)/tgamma(l1-l2);
        lam2[i] = lambda2((double) i, l1, l2 + 1.0);
        sclrow[i] = normultra2 ? sqrt(2.0*M_PI*lambda2((double) i, 2.0*l2, 1.0)/(i+l2))/(pow(2.0, l2)*tgamma(l2)) : 1.0;
        sclcol[i] = normultra1 ? sqrt((i+l1)/(2.0*M_PI*lambda2((double) i, 2.0*l1, 1.0)))*(pow(2.0, l1)*tgamma(l1)) : 1.0;
    }
    if (fabs((l1 - l2) - (int) (l1 - l2)) < M_EPS*(fabs(l1)+fabs(l2))) {
        lam1[0] = 1.0;
        for (int i = 0; i < n-1; i++)
            lam1[i+1] = (l1 - l2 + i)/(1.0 + i)*lam1[i];
    }
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2)
            A(i,j) = sclrow[i]*scl*(i+l2)*lam1[(j-i)/2]*lam2[(j+i)/2]*sclcol[j];
    free(lam1);
    free(lam2);
    free(sclrow);
    free(sclcol);
    return A;
}

double * plan_jac2jac(const int normjac1, const int normjac2, const int n, const double alpha, const double beta, const double gamma) {
    double * A = (double *) calloc(n * n, sizeof(double));
    double * lam1 = (double *) calloc(n, sizeof(double));
    double * lam2 = (double *) calloc(2*n, sizeof(double));
    double * sclrow = (double *) calloc(n, sizeof(double));
    double * sclcol = (double *) calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        lam1[i] = lambda2((double) i, alpha - gamma, 1.0)/tgamma(alpha - gamma);
        sclrow[i] = (2.0*i + gamma + beta + 1.0)*lambda2((double) i, gamma + beta + 1.0, beta + 1.0);
        sclcol[i] = lambda2((double) i, beta + 1.0, alpha + beta + 1.0);
    }
    if (fabs((alpha - gamma) - (int) (alpha - gamma)) < M_EPS*(fabs(alpha)+fabs(gamma))) {
        lam1[0] = 1.0;
        for (int i = 0; i < n-1; i++)
            lam1[i+1] = (alpha - gamma + i)/(1.0 + i)*lam1[i];
    }
    for (int i = 0; i < 2*n; i++)
        lam2[i] = lambda2((double) i, alpha + beta + 1.0, gamma + beta + 2.0);
    if (fabs(beta + gamma + 1.0) < M_EPS*(fabs(beta)+fabs(gamma)+1.0))
        sclrow[0] = 1.0/tgamma(beta + 1.0);
    if (fabs(alpha + beta + 1.0) < M_EPS*(fabs(alpha)+fabs(beta)+1.0)) {
        lam2[0] = 1.0/(sclrow[0]*lam1[0]);
        sclcol[0] = 1.0;
    }
    for (int i = 0; i < n; i++) {
        sclrow[i] *= normjac2 ? sqrt(Analphabeta(i, gamma, beta)) : 1.0;
        sclcol[i] /= normjac1 ? sqrt(Analphabeta(i, alpha, beta)) : 1.0;
    }
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i--)
            A(i,j) = sclrow[i]*lam1[j-i]*lam2[j+i]*sclcol[j];
    free(lam1);
    free(lam2);
    free(sclrow);
    free(sclcol);
    return A;
}

double * eigenplan_jac2jac(const int normjac1, const int normjac2, const int n, const double alpha, const double beta, const double gamma, const double delta) {
    triangular_banded * A = malloc_triangular_banded(n, 2);
    triangular_banded * B = malloc_triangular_banded(n, 2);
    double v;
    /*
    if (n > 0)
        set_triangular_banded_index(A, 0, 0, 0);
    for (int i = 1; i < n; i++) {
        v = i*(i+alpha+beta+1)*(i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2);
        set_triangular_banded_index(A, v, i, i);
    }
    for (int i = 0; i < n-1; i++) {
        v = (gamma-delta)*(i+gamma+delta+2)/(2*i+gamma+delta+2)/(2*i+gamma+delta+4)*((i+1)*(i+gamma+delta+2)+(gamma+delta+2)*(gamma+delta-alpha-beta)/2) - (i+gamma+delta+2)*(gamma-alpha+beta-delta)/2;
        set_triangular_banded_index(A, v, i, i+1);
    }
    for (int i = 0; i < n-2; i++) {
        v = -(i+gamma+delta+3)*(i+gamma+2)/(2*i+gamma+delta+4)*(i+delta+2)/(2*i+gamma+delta+5)*(i+gamma-alpha+delta-beta+2);
        set_triangular_banded_index(A, v, i, i+2);
    }
    if (n > 0)
        set_triangular_banded_index(B, 1, 0, 0);
    for (int i = 1; i < n; i++) {
        v = (i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2);
        set_triangular_banded_index(B, v, i, i);
    }
    for (int i = 0; i < n-1; i++) {
        v = (gamma-delta)*(i+gamma+delta+2)/(2*i+gamma+delta+2)/(2*i+gamma+delta+4);
        set_triangular_banded_index(B, v, i, i+1);
    }
    for (int i = 0; i < n-2; i++) {
        v = -(i+gamma+2)/(2*i+gamma+delta+4)*(i+delta+2)/(2*i+gamma+delta+5);
        set_triangular_banded_index(B, v, i, i+2);
    }
    */
    if (n > 0)
        set_triangular_banded_index(A, 0, 0, 0);
    if (n > 1) {
        v = (gamma+delta+2)*((gamma-delta)/(gamma+delta+4)*(1+(gamma+delta-alpha-beta)/2) - (gamma-alpha+beta-delta)/2);
        set_triangular_banded_index(A, v, 0, 1);
    }
    if (n > 2) {
        v = -(gamma+delta+3)*(gamma+2)/(gamma+delta+4)*(delta+2)/(gamma+delta+5)*(gamma-alpha+delta-beta+2);
        set_triangular_banded_index(A, v, 0, 2);
    }
    for (int i = 1; i < n; i++) {
        v = i*(i+alpha+beta+1);
        set_triangular_banded_index(A, v, i, i);
    }
    for (int i = 1; i < n-1; i++) {
        v = ((gamma-delta)/(2*i+gamma+delta+2)/(2*i+gamma+delta+4)*((i+1)*(i+gamma+delta+2)+(gamma+delta+2)*(gamma+delta-alpha-beta)/2) - (gamma-alpha+beta-delta)/2) / ((i+gamma+delta+1)/(2*i+gamma+delta+1)/(2*i+gamma+delta+2));
        set_triangular_banded_index(A, v, i, i+1);
    }
    for (int i = 1; i < n-2; i++) {
        v = -(i+gamma+delta+3)*(i+gamma+2)/(2*i+gamma+delta+4)*(i+delta+2)/(2*i+gamma+delta+5)*(i+gamma-alpha+delta-beta+2) / ((i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2));
        set_triangular_banded_index(A, v, i, i+2);
    }
    if (n > 0)
        set_triangular_banded_index(B, 1, 0, 0);
    if (n > 1) {
        v = (gamma-delta)/(gamma+delta+4);
        set_triangular_banded_index(B, v, 0, 1);
    }
    if (n > 2) {
        v = -(gamma+2)/(gamma+delta+4)*(delta+2)/(gamma+delta+5);
        set_triangular_banded_index(B, v, 0, 2);
    }
    for (int i = 1; i < n; i++) {
        set_triangular_banded_index(B, 1, i, i);
    }
    for (int i = 1; i < n-1; i++) {
        v = (gamma-delta)/(i+gamma+delta+1)*(2*i+gamma+delta+1)/(2*i+gamma+delta+4);
        set_triangular_banded_index(B, v, i, i+1);
    }
    for (int i = 1; i < n-2; i++) {
        v = -(i+gamma+2)/(2*i+gamma+delta+4)*(i+delta+2)/(2*i+gamma+delta+5) /( (i+gamma+delta+1)/(2*i+gamma+delta+1)*(i+gamma+delta+2)/(2*i+gamma+delta+2) );
        set_triangular_banded_index(B, v, i, i+2);
    }

    double * V = (double *) calloc(n*n, sizeof(double));
    if (n > 0)
        V[0] = 1.0;
    if (n > 1)
        V[1+n] = (alpha+beta+2)/(gamma+delta+2);
    //for (int i = 2; i < n; i++)
    //    V[i+i*n] = (2*i+alpha+beta-1)/(i+alpha+beta)*(2*i+alpha+beta)/(2*i+gamma+delta-1)*(i+gamma+delta)/(2*i+gamma+delta)*V[i-1+(i-1)*n];
    for (int i = 2; i < n; i++)
        V[i+i*n] = lambda2(2*i+1, alpha+beta, gamma+delta)*lambda2(i+1, gamma+delta, alpha+beta);

    triangular_banded_eigenvectors(A, B, V);

    double * sclrow = (double *) calloc(n, sizeof(double));
    double * sclcol = (double *) calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        sclrow[i] = normjac2 ? sqrt(Analphabeta(i, gamma, delta)) : 1.0;
        sclcol[i] = normjac1 ? 1.0/sqrt(Analphabeta(i, alpha, beta)) : 1.0;
    }
    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i--)
            V[i+j*n] *= sclrow[i]*sclcol[j];

    destroy_triangular_banded(A);
    destroy_triangular_banded(B);
    free(sclrow);
    free(sclcol);

    return V;
}
