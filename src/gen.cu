#include <curand.h>
#include <curand_kernel.h>

#include "common.h"
#include "type.h"

#define TPB 512
#define GET_ID() (threadIdx.x+blockDim.x*blockIdx.x)
#define MAX(a,b) ((a)>(b)?(a):(b))
#define CUFREE(x) {if(x)cudaFree(x);}


static inline void get_gpulen(const int n, int *const gpulen)
{
  if (n > TPB*512)
    *gpulen = TPB*512;
  else
    *gpulen = (int) n;
}



__global__ void setup_curand_rng(const int seed, curandState *state, const int gpulen)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  curand_init(seed, idx, 0, state + idx);
}



__global__ void runif_kernel(curandState *state, const REAL min, const REAL max, const int gpulen, REAL *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  REAL tmp = curand_uniform(state + idx);
  x[idx] = min + (max - min)*tmp;
}



int runif(const unsigned int seed, const int n, const REAL min, const REAL max, REAL *x)
{
  int gpulen;
  curandState *state;
  
  get_gpulen(n, &gpulen);
  cudaMalloc(&state, gpulen*sizeof(*state));
  if (state == NULL)
    return ERR_CUMALLOC;
  
  int runs = (int) MAX((int) n/gpulen, 1);
  int rem = (int) MAX((n - (int)(runs*gpulen)), 0);
  int runlen = MAX(gpulen/TPB, 1);
  
  setup_curand_rng<<<runlen, TPB>>>(seed, state, gpulen);
  for (int i=0; i<runs; i++)
    runif_kernel<<<runlen, TPB>>>(state, min, max, gpulen, x);
  
  if (rem)
  {
    runlen = MAX(rem/TPB, 1);
    runif_kernel<<<runlen, TPB>>>(state, min, max, gpulen, x);
  }
  
  cudaFree(state);
  
  return ERR_OK;
}



int gen_setup(const int m_local, const int n, REAL **x, REAL **y, REAL **z)
{
  cudaMalloc(x, m_local*n*sizeof(**x));
  cudaMalloc(y, n*sizeof(**y));
  cudaMalloc(z, m_local*sizeof(**z));
  
  if (*x == NULL || *y == NULL || *z == NULL)
    return ERR_CUMALLOC;
  
  return ERR_OK;
}
