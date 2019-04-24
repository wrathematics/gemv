MPICXX = mpic++
NVCC = /usr/local/cuda/bin/nvcc

CXXFLAGS = -O2 -std=c++11 -Wall -pedantic -Wextra
LDFLAGS = 
CUDACFLAGS = -arch=sm_61 -Xcompiler -Ofast
CUDACPPFLAGS = -Xcompiler -Wall -Xcompiler -Wextra
CUDALDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcublas


MPI_SRC = $(wildcard src/*.cxx)
MPI_OBJECTS = $(MPI_SRC:.cxx=.o)

CUDA_SRC = $(wildcard src/*.cu)
CUDA_OBJECTS = $(CUDA_SRC:.cu=.o)

OBJECTS = $(MPI_OBJECTS) $(CUDA_OBJECTS)

%.o: %.cxx
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDACPPFLAGS) $(CUDACFLAGS) -c $< -o $@

all: clean main

main: $(OBJECTS)
	$(MPICXX) $^ -o gemv $(CUDALDFLAGS) $(LDFLAGS)

clean:
	rm -rf gemv ./src/*.o
