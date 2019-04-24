#include <cuda_runtime.h>

#include "type.h"

#define MB_BACKOFF 1000


int get_n()
{
  size_t mem_free, mem_total;
  
  cudaSetDevice(0);
  
  cudaMemGetInfo(&mem_free, &mem_total);
  return (int) sqrt((mem_total - MB_BACKOFF*1024*1024)/sizeof(REAL));
}



void gpu_to_host(void *x_cpu, void *x_gpu, size_t len)
{
  cudaMemcpy(x_cpu, x_gpu, len, cudaMemcpyDeviceToHost);
}



void gpu_free(void *x)
{
  if (x)
    cudaFree(x);
}
