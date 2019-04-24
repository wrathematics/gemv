#ifndef GEMV_GEN_H
#define GEMV_GEN_H

#include "type.h"

void runif(const unsigned int seed, const int n, const REAL min, const REAL max, REAL *x);
int gen_setup(const int m_local, const int n, REAL **x, REAL **y, REAL **z);

#endif
