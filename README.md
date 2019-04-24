# gemv

A 1-d distributed gemv (dense matrix time vector) benchmark with gpus. The local matrix dimensions are chosen to pack the largest square matrix into gpu ram as possible.


## Compiling

You may need to set `MPICXX` in `Makefile.in`, e.g. to `CC` on a Cray.
