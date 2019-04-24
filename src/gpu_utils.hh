#ifndef GEMV_GPU_UTILS_H
#define GEMV_GPU_UTILS_H

#include <stdlib.h>

int get_n();
void gpu_to_host(void *x_cpu, void *x_gpu, size_t len);
void gpu_free(void *x);

#endif
