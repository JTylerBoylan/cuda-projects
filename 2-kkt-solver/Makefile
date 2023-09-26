# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3

# Include and Library paths
INCLUDE_PATH = -I/usr/local/include
LIBRARY_PATH = -L/usr/local/lib

# Libraries to link against
LIBRARIES = -lginac -lcln

# Source and Object files
SRC = kkt_solver.cu
OBJ = main.o
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(LIBRARY_PATH) -o $@ $^ $(LIBRARIES)

$(OBJ): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_PATH) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
