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
LIBRARIES = -lginac -lcln

# Source and Object files
SOLVER_SRC = $(wildcard *.cu)
GENERATE_SRC = $(wildcard *.cpp)

SOLVER_OBJ = $(SOLVER_SRC:.cu=.o)
GENERATE_OBJ = $(GENERATE_SRC:.cpp=.o)

all: solver generator

solver: $(SOLVER_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(LIBRARY_PATH) -o $@ $^ $(LIBRARIES)

generator: $(GENERATE_OBJ)
	$(CXX) $(CXX_FLAGS) $(LIBRARY_PATH) -o $@ $^ $(LIBRARIES)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE_PATH) -c $< -o $@

clean:
	rm -f $(SOLVER_OBJ) $(GENERATE_OBJ) $(SOLVER_TARGET) $(GENERATE_TARGET)
