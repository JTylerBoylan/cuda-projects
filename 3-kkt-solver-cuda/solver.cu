#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <assert.h>

#include "GENERATED_LOOKUP.cu"

#define NUM_ITERATIONS 50
#define TOLERANCE 1E-4

inline cudaError_t checkCuda(cudaError_t result);

__device__
float squared_sum(float * w)
{
  float sum = 0;
  for (int v = 0; v < NUM_VARIABLES; v++)
  {
    sum += w[v]*w[v];
  }
  return sum;
}

__global__
void solve(float * w, float * cost, bool * sol)
{
  assert(NUM_VARIABLES == blockDim.x);

  // Variables local to the block
  __shared__ float wi[NUM_VARIABLES];
  __shared__ float diffi[NUM_VARIABLES];
  __shared__ bool solved;

  // Indices
  const int varIdx = threadIdx.x;
  const int objIdx = blockIdx.x;
  const int globalIdx = objIdx*NUM_VARIABLES + varIdx;

  // Initialize
  if (varIdx == 0)
  {
    solved = false;
  }
  __syncthreads();

  // Get from global lookup function
  wi[varIdx] = LOOKUP_INITIAL[globalIdx];

  // Run Newton-Raphson
  for (int iter = 0; iter < NUM_ITERATIONS; iter++)
  {

    // Break if solved
    if (solved) break;

    // Evaluate from global lookup function
    diffi[varIdx] = LOOKUP_INTERCEPT[globalIdx](wi);

    // Apply
    wi[varIdx] -= diffi[varIdx];

    // Check if solved
    if (varIdx == 0 && squared_sum(diffi) < TOLERANCE)
    {
      solved = true;
    }

    // Make sure the entire block is done before iterating
    __syncthreads();
  }

  // Save results
  w[globalIdx] = wi[varIdx];
  if (varIdx == 0)
  {
    cost[objIdx] = COST(wi);
    sol[objIdx] = solved;
  }

}

#define CYCLES 10000L

int main()
{

  // Allocate
  float * w;
  float * cost;
  bool * sol;
  checkCuda( cudaMallocManaged(&w, NUM_OBJECTIVES*NUM_VARIABLES*sizeof(float)) );
  checkCuda( cudaMallocManaged(&cost, NUM_OBJECTIVES*sizeof(float)) );
  checkCuda( cudaMallocManaged(&sol, NUM_OBJECTIVES*sizeof(bool)) );

  // Solve
  auto cstart = std::chrono::high_resolution_clock::now();
  for (int cyc = 0; cyc < CYCLES; cyc++)
  {
    solve<<<NUM_OBJECTIVES, NUM_VARIABLES>>>(w, cost, sol);
    checkCuda( cudaDeviceSynchronize() );
  }
  auto cend = std::chrono::high_resolution_clock::now();

  time_t time_us = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count();
  printf("Cycles: %lu, Time: %lu us\n", CYCLES, time_us);
  printf("Performance: %lu cycles/s\n", CYCLES*(1000000L)/time_us);

  float solution[] = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0};

  // Print Results
  float squared_sum_err = 0.0f;
  for (int i = 0; i < NUM_OBJECTIVES*NUM_VARIABLES; i++)
  {
    squared_sum_err += (w[i] - solution[i])*(w[i] - solution[i]);
    printf("W(%d) = %f ", i, w[i]);
    printf("%s ", sol[i/NUM_VARIABLES] ? "(solved)" : "(unsolved)");
    printf("cost = %f\n", cost[i/NUM_VARIABLES]);
  }
  printf("Error: %f\n", squared_sum_err);

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