#include <cublas_v2.h>

#include "common.h"
#include "restrict.h"
#include "type.h"

// ----------------------------------------------------------------------------
// c    = A    *    b
//  mx1    mxn       nx1
// ----------------------------------------------------------------------------

int mvm_gemm(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const REAL alpha = 1.0;
  const REAL beta = 0.0;
  const int k = 1;
  
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate_v2(&handle);
  if (st != CUBLAS_STATUS_SUCCESS)
    return ERR_CUBLAS;
  
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
#if TYPE == FLOAT
  cublasSgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
#elif TYPE == DOUBLE
  cublasDgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
#endif
  cublasDestroy_v2(handle);
  
  return ERR_OK;
}



int mvm_gemv(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const REAL alpha = 1.0;
  const REAL beta = 0.0;
  const int inc = 1;
  
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate_v2(&handle);
  if (st != CUBLAS_STATUS_SUCCESS)
    return ERR_CUBLAS;
  
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
#if TYPE == FLOAT
  cublasSgemv(handle, trans, m, n, &alpha, A, m, b, inc, &beta, c, inc);
#elif TYPE == DOUBLE
  cublasDgemv(handle, trans, m, n, &alpha, A, m, b, inc, &beta, c, inc);
#endif
  cublasDestroy_v2(handle);
  
  return ERR_OK;
}
