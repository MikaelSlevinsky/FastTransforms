#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

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

#define A(i,j) A[(i)+n*(j)]

void printmat(char * MAT, char * FMT, double * A, int n, int m) {
    printf("%s = \n", MAT);
    if (n > 0 && m > 0) {
        if (signbit(A(0,0))) {printf("[");}
        else {printf("[ ");}
        printf(FMT, A(0,0));
        for (int j = 1; j < m; j++) {
            if (signbit(A(0,j))) {printf("  ");}
            else {printf("   ");}
            printf(FMT, A(0,j));
        }
        for (int i = 1; i < n-1; i++) {
            printf("\n");
            if (signbit(A(i,0))) {printf(" ");}
            else {printf("  ");}
            printf(FMT, A(i,0));
            for (int j = 1; j < m; j++) {
                if (signbit(A(i,j))) {printf("  ");}
                else {printf("   ");}
                printf(FMT, A(i,j));
            }
        }
        if (n > 1) {
            printf("\n");
            if (signbit(A(n-1,0))) {printf(" ");}
            else {printf("  ");}
            printf(FMT, A(n-1,0));
            for (int j = 1; j < m; j++) {
                if (signbit(A(n-1,j))) {printf("  ");}
                else {printf("   ");}
                printf(FMT, A(n-1,j));
            }
        }
        printf("]\n");
    }
}

int main(void) {
    double z = 10.0;
    double s = stirlingseries(z);

    printf("Hello, World!\n");
    printf("The Stirling series evaluated at z = %3.1f is s(z) = %17.16e.\n", z, s);

    int n = 10;

    double * A = (double *) calloc(n*n, sizeof(double));
    double * B = (double *) calloc(n*n, sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A[i+j*n] = B[i+j*n] = 1.0/(i+j*n+1.0);

    printmat("A", "%1.3f", A, n, n);
    printmat("B", "%1.3f", A, n, n);

    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1.0, A, n, B, n);

    printmat("triu(A)*B", "%1.3f", B, n, n);

    return 0;
}
