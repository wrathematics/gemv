#include <cuda_runtime.h>

#include "type.h"

#define MB_BACKOFF 1000


void gpu_init()
{
  cudaSetDevice(0);
  cudaDeviceReset();
}



int get_dims()
{
  size_t mem_free, mem_total;
  
  cudaMemGetInfo(&mem_free, &mem_total);
  
  double bytes = (double)(mem_total - (size_t)MB_BACKOFF*1024*1024);
  return (int) sqrt(bytes/sizeof(REAL));
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
