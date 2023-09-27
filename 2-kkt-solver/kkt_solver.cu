#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <assert.h>
#include <ginac/ginac.h>

#define TPB 256

using namespace GiNaC;

// f(x) = cost function
ex objective_function(std::vector<symbol>& x)
{
  return x[0]*x[0] + x[1]*x[1];
}

// g(x) <= 0
exvector inequality_constraints(std::vector<symbol>& x)
{
  return {
    x[0] - 1,
    x[1] - 0,
    -x[0] + 1
  };
}

// Set initial values for x, lambda, and s
void initialize(std::vector<float>& x0, std::vector<float>& lambda0, std::vector<float>& s0)
{
  x0 = {-0.9f, -0.1f};
  lambda0 = {1.0f, 1.0f, 1.0f};
  s0 = {1.0f, 1.0f, 1.0f};
}



inline cudaError_t checkCuda(cudaError_t result);

void create_index_vector(std::vector<symbol>& vec, const int size, const std::string var);

exvector diff_function_by_vec(const ex& func, const std::vector<symbol>& vec);

int main()
{

  // Get initial vectors
  std::vector<float> x0, lambda0, s0;
  initialize(x0, lambda0, s0);

  printf("Initial vectors created.\n");

  // Get sizes of states and constraints
  const size_t x_len = x0.size();
  const size_t lambda_len = lambda0.size();
  const size_t s_len = s0.size();

  // Check if vectors are valid sizes
  if (lambda_len != s_len)
  {
    printf("Incorrect size error: Lambda vector size (%lu) must equal S vector size (%lu).", 
            lambda_len, s_len);
    assert(lambda_len == s_len);
  }

  // Create expression vectors
  std::vector<symbol> x, lambda, s;
  create_index_vector(x, x_len, "x");
  create_index_vector(lambda, lambda_len, "h");
  create_index_vector(s, s_len, "s");

  printf("Expression vectors created.\n");

  // Get ineq. constraint vector
  exvector g = inequality_constraints(x);

  // Check for valid vector sizes
  const size_t g_size = g.size();
  if (g_size != lambda_len)
  {
    printf("Incorrect size error: Inequality constraints vector size (%lu) must equal lambda vector size (%lu).\n",
            g_size, lambda_len);
    assert(g_size == lambda_len);
  }

  // Lagrange equation
  ex lagrange_equation = objective_function(x);
  for (int l = 0; l < lambda_len; l++)
  {
    lagrange_equation += lambda[l]*(g[l] + s[l]*s[l]);
  }

  std::cout << "L(x, h, s) = " << lagrange_equation << std::endl;

  // KKT vector
  const exvector dLdx = diff_function_by_vec(lagrange_equation, x);
  const exvector dLdh = diff_function_by_vec(lagrange_equation, lambda);
  const exvector dLds = diff_function_by_vec(lagrange_equation, s);
  exvector KKT = dLdx;
  KKT.insert(KKT.end(), dLdh.begin(), dLdh.end());
  KKT.insert(KKT.end(), dLds.begin(), dLds.end());

  for (int k = 0; k < KKT.size(); k++)
  {
    std::cout << "K(" << k << ") = " << KKT[k] << std::endl;
  }

  /*TODO*/

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

// Fill an expression vector with indexed variables
void create_index_vector(std::vector<symbol>& vec, const int size, const std::string var)
{
  vec.reserve(size);
  for (int i = 0; i < size; i++)
  {
    std::string var_i = var + std::to_string(i);
    symbol xi(var_i);
    vec.push_back(xi);
  }
}

// Get the derivative of a function with respect to a vector
exvector diff_function_by_vec(const ex& func, const std::vector<symbol>& vec)
{
  const size_t len = vec.size();
  exvector jacobian;
  jacobian.reserve(len);
  for (int i = 0; i < len; i++)
  {
    jacobian.push_back(func.diff(vec[i]));
  }
  return jacobian;
}