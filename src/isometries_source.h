FLT * X(Gx)(int l);
FLT * X(Gy)(int l);
FLT * X(Gz)(int l);
FLT X(Gy_index)(int l, int i, int j);
FLT X(Gz_index)(int l, int i, int j);
FLT X(Gzhat_index)(int l, int i, int j);
FLT X(Gzhatinv_index)(int l, int i, int j);
FLT X(X_index)(int l, int alpha, int i, int j);
FLT X(Y_index)(int l, int i, int j);
FLT X(Z_index)(int l, int i, int j);
FLT X(Z_diagonal)(int l, int i, int j);

int X(is_J_entry_nonzero)(int l, int i, int j);
FLT * X(J)(int l);
FLT * X(J_eigen)(int l);
FLT X(XJXJ)(FLT * J, int l, int i, int j, FLT beta, FLT gamma);

FLT * X(rotation_matrix_J)(int l, FLT alpha, FLT beta, FLT gamma, FLT * J);
FLT * X(rotation_matrix_direct)(int l, FLT alpha, FLT beta, FLT gamma);
FLT * X(rotation_matrix)(int l, FLT alpha, FLT beta, FLT gamma);
