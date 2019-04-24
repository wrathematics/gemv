#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>


void MPI_print(int rank, const char *fmt, ...)
{
  if (rank == 0)
  {
    va_list args;
    
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
  }
}



void MPI_throw_err(int errno, int rank, const char *fmt, ...)
{
  if (rank == 0)
  {
    va_list args;
    
    fprintf(stderr, "ERROR: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
  }
  
  exit(errno);
}
