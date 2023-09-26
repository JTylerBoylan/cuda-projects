#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <assert.h>
#include <ginac/ginac.h>

#define TPB 256

using namespace GiNaC;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

int main()
{

    symbol x("x");
    ex poly = pow(x,2) + 3*x + 2;

    std::cout << "Polynomial: " << poly << std::endl;

    return 0;
}