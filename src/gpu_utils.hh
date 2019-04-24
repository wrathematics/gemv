#ifndef GEMV_GPU_UTILS_H
#define GEMV_GPU_UTILS_H

#include <stdlib.h>

void gpu_init();
int get_dims();
void gpu_to_host(void *x_cpu, void *x_gpu, size_t len);
void gpu_free(void *x);

#endif
