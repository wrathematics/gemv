#ifndef GEMV_MPI_UTILS_H
#define GEMV_MPI_UTILS_H


#include <mpi.h>
#include <stdarg.h>

void allreduce(void *x, int len, MPI_Comm comm);
void MPI_print(int rank, const char *fmt, ...);
void MPI_check_err(int errno, int rank, const char *fmt, ...);


#endif
