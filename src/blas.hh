#ifndef GEMV_BLAS_H
#define GEMV_BLAS_H

#include "restrict.h"
#include "type.h"

int mvm_gemm(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c);
  
int mvm_gemv(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c);

#endif
