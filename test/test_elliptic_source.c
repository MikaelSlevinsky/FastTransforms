void Y(test_elliptic)(int * checksum) {
    printf("\t\t\t Test \t\t\t\t | 2-norm Relative Error\n");
    printf("---------------------------------------------------------|----------------------\n");

    FLT k = 1/Y(sqrt)(2);
    FLT K = X(complete_elliptic_integral)('1', k);
    FLT err = Y(fabs)(K - Y(pow)(Y(tgamma)(0.25), 2)/(4*Y(tgamma)(0.5)))/Y(fabs)(Y(pow)(Y(tgamma)(0.25), 2)/(4*Y(tgamma)(0.5)));
    printf("Complete elliptic integral, lemniscatic case, k²= %3.1f \t |%20.2e ", (double) (k*k), (double) err);
    X(checktest)(err, 2, checksum);

    k = Y(__sinpi)(ONE(FLT)/12);
    FLT kp = Y(sqrt)((1-k)*(1+k));
    K = X(complete_elliptic_integral)('1', k);
    FLT E = X(complete_elliptic_integral)('2', k);
    err = Y(fabs)(E - Y(atan)(1)/(Y(sqrt)(3)*K) - Y(sqrt)(TWO(FLT)/3)*kp*K)/Y(fabs)(E);
    printf("Complete elliptic integrals, first and second kinds \t |%20.2e ", (double) err);
    X(checktest)(err, 2, checksum);

    k = ((FLT) 3)/5;
    kp = Y(sqrt)((1-k)*(1+k));
    K = X(complete_elliptic_integral)('1', k);
    E = X(complete_elliptic_integral)('2', k);
    FLT KP = X(complete_elliptic_integral)('1', kp);
    FLT EP = X(complete_elliptic_integral)('2', kp);
    err = Y(fabs)(E*KP+EP*K-K*KP-2*Y(atan)(1))/(2*Y(atan)(1));
    printf("Legendre relation EK'+E'K-KK' = π/2,          k = %3.1f \t |%20.2e ", (double) k, (double) err);
    X(checktest)(err, 2, checksum);

    FLT sn, cn, dn;
    K = X(complete_elliptic_integral)('1', k);
    X(jacobian_elliptic_functions)(0*K, k, &sn, &cn, &dn, FT_SN | FT_CN | FT_DN);
    err = Y(fabs)(sn-0) + Y(fabs)(cn-1) + Y(fabs)(dn-1);
    printf("Jacobian elliptic functions at x = %3.1fK(%3.1f), k = %3.1f \t |%20.2e ", 0.0, (double) k, (double) k, (double) err);
    X(checktest)(err, 2, checksum);

    X(jacobian_elliptic_functions)(K/2, k, &sn, &cn, &dn, FT_SN | FT_CN | FT_DN);
    err = Y(fabs)(sn-1/Y(sqrt)(1+kp)) + Y(fabs)(cn-Y(sqrt)(kp/(1+kp))) + Y(fabs)(dn-Y(sqrt)(kp));
    printf("Jacobian elliptic functions at x = %3.1fK(%3.1f), k = %3.1f \t |%20.2e ", 0.5, (double) k, (double) k, (double) err);
    X(checktest)(err, 2, checksum);

    X(jacobian_elliptic_functions)(1*K, k, &sn, &cn, &dn, FT_SN | FT_CN | FT_DN);
    err = Y(fabs)(sn-1) + Y(fabs)(cn-0) + Y(fabs)(dn-kp);
    printf("Jacobian elliptic functions at x = %3.1fK(%3.1f), k = %3.1f \t |%20.2e ", 1.0, (double) k, (double) k, (double) err);
    X(checktest)(err, 2, checksum);

    X(jacobian_elliptic_functions)(3*K/2, k, &sn, &cn, &dn, FT_SN | FT_CN | FT_DN);
    err = Y(fabs)(sn-1/Y(sqrt)(1+kp)) + Y(fabs)(cn+Y(sqrt)(kp/(1+kp))) + Y(fabs)(dn-Y(sqrt)(kp));
    printf("Jacobian elliptic functions at x = %3.1fK(%3.1f), k = %3.1f \t |%20.2e ", 1.5, (double) k, (double) k, (double) err);
    X(checktest)(err, 3, checksum);

    X(jacobian_elliptic_functions)(2*K, k, &sn, &cn, &dn, FT_SN | FT_CN | FT_DN);
    err = Y(fabs)(sn-0) + Y(fabs)(cn+1) + Y(fabs)(dn-1);
    printf("Jacobian elliptic functions at x = %3.1fK(%3.1f), k = %3.1f \t |%20.2e ", 2.0, (double) k, (double) k, (double) err);
    X(checktest)(err, 3, checksum);

    FLT x = 1.0;
    FLT snx, cnx, dnx;
    X(jacobian_elliptic_functions)(x, k, &snx, &cnx, &dnx, FT_SN | FT_CN | FT_DN);
    FLT y = 2.0;
    FLT sny, cny, dny;
    X(jacobian_elliptic_functions)(y, k, &sny, &cny, &dny, FT_SN | FT_CN | FT_DN);
    FLT xy = x+y;
    FLT snxy, cnxy, dnxy;
    X(jacobian_elliptic_functions)(x+y, k, &snxy, &cnxy, &dnxy, FT_SN | FT_CN | FT_DN);
    err = Y(fabs)(snxy-(snx*cny*dny+sny*cnx*dnx)/(1-k*k*snx*snx*sny*sny))/Y(fabs)(snxy);
    err += Y(fabs)(cnxy-(cnx*cny-snx*sny*dnx*dny)/(1-k*k*snx*snx*sny*sny))/Y(fabs)(cnxy);
    err += Y(fabs)(dnxy-(dnx*dny-k*k*snx*sny*cnx*cny)/(1-k*k*snx*snx*sny*sny))/Y(fabs)(dnxy);
    printf("Jacobian addition theorems, x = %3.1f, y = %3.1f, k = %3.1f \t |%20.2e ", (double) x, (double) y, (double) k, (double) err);
    X(checktest)(err, 16, checksum);
}
