# Compilers
NVCC = nvcc
CXX = g++

# Compiler flags
NVCC_FLAGS = -O3
CXX_FLAGS = -O3

# Include and Library paths
INCLUDE_PATH = -I/usr/local/include
LIBRARY_PATH = -L/usr/local/lib

# Libraries to link against
NVCC_LIBRARIES = -lcublas -lcusolver -lcudart -lcurand
CXX_LIBRARIES = -lginac -lcln


# Targets
all: generator solver

generator: generator.o
	$(CXX) $(CXX_FLAGS) $(LIBRARY_PATH) -o generator generator.o $(CXX_LIBRARIES)

solver: solver.o
	$(NVCC) $(NVCC_FLAGS) $(LIBRARY_PATH) -o solver solver.o $(NVCC_LIBRARIES)

generator.o: generate_cuda.cpp generate_util.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_PATH) -c generate_cuda.cpp -o generator.o

solver.o: solver.cu GENERATED_LOOKUP.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c solver.cu -o solver.o

# Custom targets to build and run
generate: generator
	./generator

solve: solver
	./solver

run: generate solve

clean:
	rm -f solver generator *.o
