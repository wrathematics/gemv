#include <cublas_v2.h>

#include "common.h"
#include "restrict.h"
#include "type.h"

cublasOperation_t trans;
int k;
int inc;
cublasHandle_t handle;
REAL *alpha;
REAL *beta;


int mvm_init()
{
  trans = CUBLAS_OP_N;
  k = 1;
  inc = 1;
  
  const REAL alpha_cpu = (REAL) 1.0;
  const REAL beta_cpu = (REAL) 0.0;
  cudaMalloc(&alpha, sizeof(*alpha));
  cudaMalloc(&beta, sizeof(*beta));
  if (alpha == NULL || beta == NULL)
    return ERR_CUMALLOC;
  
  cudaMemcpy(alpha, &alpha_cpu, 1, cudaMemcpyHostToDevice);
  cudaMemcpy(beta, &beta_cpu, 1, cudaMemcpyHostToDevice);
  
  cublasStatus_t st = cublasCreate_v2(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  
  if (st != CUBLAS_STATUS_SUCCESS)
    return ERR_CUBLAS;
  else
    return ERR_OK;
}



void mvm_cleanup()
{
  cudaFree(alpha);
  cudaFree(beta);
  cublasDestroy_v2(handle);
}



// ----------------------------------------------------------------------------
// c    = A    *    b
//  mx1    mxn       nx1
// ----------------------------------------------------------------------------

void mvm_gemm(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c)
{
#if TYPE == FLOAT
  cublasSgemm(handle, trans, trans, m, k, n, alpha, A, m, b, n, beta, c, m);
#elif TYPE == DOUBLE
  cublasDgemm(handle, trans, trans, m, k, n, alpha, A, m, b, n, beta, c, m);
#endif
}



void mvm_gemv(const int m, const int n, const REAL *const restrict A,
  const REAL *const restrict b, REAL *const restrict c)
{
#if TYPE == FLOAT
  cublasSgemv(handle, trans, m, n, alpha, A, m, b, inc, beta, c, inc);
#elif TYPE == DOUBLE
  cublasDgemv(handle, trans, m, n, alpha, A, m, b, inc, beta, c, inc);
#endif
}
