#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <assert.h>

#include "GENERATED_LOOKUP.cu"

#define NUM_ITERATIONS 20
#define TOLERANCE 1E-9
#define EPSILON 1E-6

inline cudaError_t checkCuda(cudaError_t result);

__global__
void solve(float * w, float * cost, bool * sol)
{
  assert(NUM_VARIABLES == blockDim.x);

  // Variables local to the block
  __shared__ float wi[NUM_VARIABLES];
  __shared__ bool soli[NUM_VARIABLES];

  // Indices
  const int varIdx = threadIdx.x;
  const int globalIdx = blockIdx.x*NUM_VARIABLES + varIdx;

  // Get from global lookup function
  wi[varIdx] = LOOKUP_INITIAL[globalIdx];

  // Run Newton-Raphson
  for (int iter = 0; iter < NUM_ITERATIONS; iter++)
  {
    if (!soli[varIdx]) {

      // Evaluate from global lookup function
      const int diff = LOOKUP_INTERCEPT[globalIdx](wi);

      // Apply
      wi[varIdx] -= diff;

      // Check if solved
      if (diff < TOLERANCE)
      {
        soli[varIdx] = true;
      }
    }
    // Make sure the entire block is done before iterating
    __syncthreads();
  }

  // Save results
  w[globalIdx] = wi[varIdx];
  cost[globalIdx] = COST(wi);
  sol[globalIdx] = soli[varIdx];

}

int main()
{

  // Allocate
  float * w;
  float * cost;
  bool * sol;
  checkCuda( cudaMallocManaged(&w, NUM_OBJECTIVES*NUM_VARIABLES*sizeof(float)) );
  checkCuda( cudaMallocManaged(&cost, NUM_OBJECTIVES*NUM_VARIABLES*sizeof(float)) );
  checkCuda( cudaMallocManaged(&sol, NUM_OBJECTIVES*NUM_VARIABLES*sizeof(bool)) );

  // Solve
  solve<<<NUM_OBJECTIVES, NUM_VARIABLES>>>(w, cost, sol);
  checkCuda( cudaDeviceSynchronize() );

  // Print Results
  for (int i = 0; i < NUM_OBJECTIVES*NUM_VARIABLES; i++)
  {
    printf("W(%d) = %f ", i, w[i]);
    printf("%s ", sol[i] ? "(solved)" : "(unsolved)");
    printf("cost = %f\n", cost[i]);
  }

  // Free
  checkCuda( cudaFree(w) );
  checkCuda( cudaFree(cost) );
  checkCuda( cudaFree(sol) );

  return 0;
}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}