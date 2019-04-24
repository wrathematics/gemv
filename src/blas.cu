#include <cublas_v2.h>

#include "common.h"
#include "restrict.h"
#include "type.h"

cublasHandle_t handle;

int mvm_init()
{
  cublasStatus_t st = cublasCreate_v2(&handle);
  if (st != CUBLAS_STATUS_SUCCESS)
    return ERR_CUBLAS;
  else
    return ERR_OK;
}



void mvm_cleanup()
{
  cublasDestroy_v2(handle);
}



// ----------------------------------------------------------------------------
// c    = A    *    b
//  mx1    mxn       nx1
// ----------------------------------------------------------------------------

void mvm_gemm(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const REAL alpha = 1.0;
  const REAL beta = 0.0;
  const int k = 1;
  
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
#if TYPE == FLOAT
  cublasSgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
#elif TYPE == DOUBLE
  cublasDgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
#endif
}



void mvm_gemv(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const REAL alpha = 1.0;
  const REAL beta = 0.0;
  const int inc = 1;
  
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
#if TYPE == FLOAT
  cublasSgemv(handle, trans, m, n, &alpha, A, m, b, inc, &beta, c, inc);
#elif TYPE == DOUBLE
  cublasDgemv(handle, trans, m, n, &alpha, A, m, b, inc, &beta, c, inc);
#endif
}
