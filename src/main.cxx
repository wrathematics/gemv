#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "blas.hh"
#include "common.h"
#include "gen.hh"
#include "gpu_utils.hh"
#include "mpi_utils.h"
#include "type.h"

#define UNIF_MIN 0
#define UNIF_MAX 1
#define SEED 1234

#define PRINT_VERBOSE

MPI_Comm comm = MPI_COMM_WORLD;
int size, rank;



static inline double gflops(const int m, const int n, const double t)
{
  return (double)((double)2*m*n - m)/t/1e9;
}



static inline void get_dims(int *restrict m, int *restrict m_local, int *restrict n)
{
  *m = *m_local = *n = get_n();
  MPI_Allreduce(MPI_IN_PLACE, m, 1, MPI_INT, MPI_SUM, comm);
}



static inline void gemv(const int m_local, const int n,
  const REAL *const restrict x, const REAL *const restrict y,
  REAL *const restrict z, REAL *const restrict z_cpu)
{
  int check;
  
  check = mvm_gemm(m_local, n, x, y, z);
  if (check != ERR_OK)
    MPI_throw_err(check, rank, "cublas error");
  
  gpu_to_host(z_cpu, z, (size_t)m_local*sizeof(*z_cpu));
  MPI_Allreduce(MPI_IN_PLACE, z_cpu, m_local, MPI_DOUBLE, MPI_SUM, comm);
}



int main()
{
  int m, m_local, n;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  get_dims(&m, &m_local, &n);
  
  REAL *x, *y, *z;
  gen_setup(m_local, n, &x, &y, &z);
  
  REAL *z_cpu = (REAL*) malloc(m_local*sizeof(*z_cpu));
  if (z_cpu == NULL)
    MPI_throw_err(ERR_HOSTMALLOC, rank, "could not allocate host memory");
  
  
  double t0_gen = MPI_Wtime();
  runif(SEED, m_local*n, UNIF_MIN, UNIF_MAX, x);
  runif(SEED, n, UNIF_MIN, UNIF_MAX, y);
  double t1_gen = MPI_Wtime();
  MPI_Barrier(comm);
  
  
  double t0_mv = MPI_Wtime();
  gemv(m_local, n, x, y, z, z_cpu);
  double t1_mv = MPI_Wtime();
  
  
  gpu_free(x);
  gpu_free(y);
  gpu_free(z);
  free(z_cpu);
  
  
  double size_gb = (double)m*n/1024/1024/1024*sizeof(REAL);
  double time_gen = t1_gen - t0_gen;
  double time_mv = t1_mv - t0_mv;
  double gflops_mv = gflops(m, n, t1_mv-t0_mv);
#ifdef PRINT_VERBOSE
  MPI_print(rank, "Matrix size:       %dx%d\n", m, n);
  MPI_print(rank, "FP bytes:          %d\n", sizeof(REAL));
  MPI_print(rank, "Problem size (GB): %.2f\n", size_gb);
  MPI_print(rank, "Gen time:          %.2f\n", time_gen);
  MPI_print(rank, "MV time:           %.2f\n", time_mv);
  MPI_print(rank, "MV GFLOPs:         %.2f\n", gflops_mv);
  MPI_print(rank, "Avg MV GFLOPs/gpu: %.2f\n", gflops_mv/size);
#else
  MPI_print(rank, "%d,%d,%d,%f,%f,%f,%f\n", size, m, n, size_gb, time_gen, time_mv, gflops_mv);
#endif
  
  MPI_Finalize();
}
