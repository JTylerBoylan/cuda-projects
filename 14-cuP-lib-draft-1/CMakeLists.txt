cmake_minimum_required(VERSION 3.10)

project(cuP_solver)

# Specify the required CUDA version
find_package(CUDA REQUIRED)

# Set CUDA architectures if necessary
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")

# Enable separable compilation
set(CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(include)

# Define the library target
cuda_add_library(cuPSolver 
    src/solve.cu
)

# Define the executable target and its source files
add_executable(cuP_test 
    test/cuP_test.cpp
)

# Link the executable against the library we defined earlier
target_link_libraries(cuP_test
    cuPSolver
)