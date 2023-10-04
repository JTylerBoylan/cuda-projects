#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <assert.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "GENERATED_LOOKUP.cu"

#define NUM_ITERATIONS 50
#define TOLERANCE 1E-2

inline cudaError_t checkCuda(cudaError_t result);

#define CYCLES 1L

int main()
{

    float * w = 0;
    checkCuda( cudaMalloc(&w, NUM_OBJECTIVES*NUM_VARIABLES*sizeof(float)) );

    float * coeffs = 0;
    checkCuda( cudaMalloc(&coeffs, NUM_OBJECTIVES*NUM_COEFFICIENTS*sizeof(float)) );

    float ** KKT = 0;
    float * d_KKT = 0;
    checkCuda( cudaMalloc(&KKT, NUM_OBJECTIVES*sizeof(float*)) );
    checkCuda( cudaMalloc(&d_KKT, NUM_OBJECTIVES*NUM_VARIABLES*sizeof(float)) );

    float ** J = 0;
    float * d_J = 0;
    checkCuda( cudaMalloc(&J, NUM_OBJECTIVES*sizeof(float*)) );
    checkCuda( cudaMalloc(&d_J, NUM_OBJECTIVES*NUM_VARIABLES*NUM_VARIABLES*sizeof(float)) );

    cublasHandle_t handle;
    cublasCreate(&handle);

    int *d_infoArray;  // Info array
    int *d_PivotArray;  // Pivot array for LU factorization
    checkCuda( cudaMalloc((void**)&d_infoArray, NUM_OBJECTIVES * sizeof(int)) );
    checkCuda( cudaMalloc((void**)&d_PivotArray, NUM_VARIABLES * NUM_OBJECTIVES * sizeof(int)) );

    // Solve
    auto cstart = std::chrono::high_resolution_clock::now();
    for (int cyc = 0; cyc < CYCLES; cyc++)
    {

        GET_WI(w);

        // GET_COEFFS(coeffs);

        for (int iter = 0; iter < NUM_ITERATIONS; iter++)
        {
          GET_KKT(d_KKT, w, coeffs);
          FORMAT_KKT(KKT, d_KKT);

          GET_J(d_J, w, coeffs);
          FORMAT_J(J, d_J);

          checkCuda( cudaDeviceSynchronize() );

          // LU Factorization
          cublasSgetrfBatched(handle, NUM_VARIABLES, J, NUM_VARIABLES, d_PivotArray, d_infoArray, NUM_OBJECTIVES);

          // Solve
          cublasSgetrsBatched(handle, CUBLAS_OP_N, NUM_VARIABLES, NUM_OBJECTIVES, (const float**) J, NUM_VARIABLES,
            d_PivotArray, KKT, NUM_VARIABLES, d_infoArray, NUM_OBJECTIVES);


          
        }

    }
    auto cend = std::chrono::high_resolution_clock::now();

    time_t time_us = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count();
    printf("Cycles: %lu, Time: %lu us\n", CYCLES, time_us);
    printf("Performance: %lu cycles/s\n", CYCLES*(1000000L)/time_us);

    cublasDestroy(handle);
    checkCuda( cudaFree(w) );
    checkCuda( cudaFree(coeffs) );
    checkCuda( cudaFree(d_KKT) );
    checkCuda( cudaFree(KKT) );
    checkCuda( cudaFree(d_J) );
    checkCuda( cudaFree(J) );

    return EXIT_SUCCESS;
}


inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}