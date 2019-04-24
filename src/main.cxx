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
#define REPLICATIONS 10

#define PRINT_VERBOSE

MPI_Comm comm = MPI_COMM_WORLD;
int size, rank;



static inline double gflops(const int m, const int n, const double t)
{
  double ops = (double) m*(2*n+2); // gemm formula
  // double ops = (double) 2.0*m*n - m; // gemv formula
  return ops/t/1e9;
}



static inline void gemv(const int m_local, const int n,
  const REAL *const restrict x, const REAL *const restrict y,
  REAL *const restrict z, REAL *const restrict z_cpu)
{
  mvm_gemm(m_local, n, x, y, z);
  gpu_to_host(z_cpu, z, (size_t)m_local*sizeof(*z_cpu));
  allreduce(z_cpu, m_local, comm);
}



int main()
{
  int check;
  int m, m_local, n;
  
  MPI_Init(NULL, NULL);
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  gpu_init();
  
  m = m_local = n = get_dims();
  MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_INT, MPI_SUM, comm);
  
  REAL *x, *y, *z;
  check = gen_setup(m_local, n, &x, &y, &z);
  MPI_check_err(check, rank, "could not allocate device memory");
  
  REAL *z_cpu = (REAL*) malloc(m_local*sizeof(*z_cpu));
  MPI_check_err((z_cpu == NULL)?ERR_HOSTMALLOC:ERR_OK, rank, "could not allocate host memory");
  
  
  double t0_gen = MPI_Wtime();
  runif(SEED, m_local*n, UNIF_MIN, UNIF_MAX, x);
  runif(SEED, n, UNIF_MIN, UNIF_MAX, y);
  double t1_gen = MPI_Wtime();
  
  check = mvm_init();
  MPI_check_err(check, rank, "cublas error");
  MPI_Barrier(comm);
  
  double t0_mv = MPI_Wtime();
  for (int i=0; i<REPLICATIONS; i++)
    gemv(m_local, n, x, y, z, z_cpu);
  double t1_mv = MPI_Wtime();
  
  mvm_cleanup();
  
  gpu_free(x);
  gpu_free(y);
  gpu_free(z);
  free(z_cpu);
  
  
  double size_gb = (double)m*n/1024/1024/1024*sizeof(REAL);
  double time_gen = t1_gen - t0_gen;
  double time_mv = (t1_mv - t0_mv) / REPLICATIONS;
  double gflops_mv = gflops(m, n, time_mv);
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
