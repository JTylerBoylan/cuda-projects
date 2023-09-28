#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <memory>
#include <assert.h>
#include <ginac/ginac.h>

#include "generate_util.cpp"

#define NUM_OBJECTIVES 1
#define NUM_STATES 2
#define NUM_CONSTRAINTS 3

#define NUM_VARIABLES NUM_STATES+2*NUM_CONSTRAINTS

using namespace GiNaC;

using symbol_ptr = std::shared_ptr<symbol>;

// f(x) = cost function
ex objective_function(std::vector<symbol_ptr>& x)
{
  return (*x[0])*(*x[0]) + (*x[1])*(*x[1]);
}

// g(x) <= 0
matrix inequality_constraints(std::vector<symbol_ptr>& x)
{
  matrix constraints(3,1);
  constraints(0,0) = (*x[0]) - 1;
  constraints(1,0) = (*x[1]) - 0;
  constraints(2,0) = -(*x[0]) + 1;
  return constraints;
}


void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var);

matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec);

matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec);


int main()
{

  // Create expression vectors
  std::vector<symbol_ptr> w;
  create_index_vector(w, NUM_STATES, "w");

  // Get ineq. constraint vector
  const matrix g = inequality_constraints(w);

  assert(g.cols() == NUM_CONSTRAINTS);

  // Lagrange equation
  ex lagrange_equation = objective_function(w);
  for (int l = 0; l < NUM_CONSTRAINTS; l++)
  {
    const symbol_ptr lambda = w[NUM_STATES+l];
    const symbol_ptr s = w[NUM_STATES+NUM_CONSTRAINTS+l];
    lagrange_equation += (*lambda)*(g(l,0) + (*s)*(*s));
  }

  // Get the jacobian of the objective w.r.t the symbols
  const matrix KKT = diff_function_by_vec(lagrange_equation, w);

  // Get the hessian of the objective w.r.t the symbols
  const matrix hessian = diff_vec_by_vec(KKT, w);

  // Get the inverse of the hessian
  const matrix inv_hessian = inverse(hessian);

  // Get the inverse hessian * KKT
  matrix inv_hess_KKT = inv_hessian.mul(KKT);

  print_j(inv_hess_KKT);

  return 0;
}

// Fill an expression vector with indexed variables
void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var)
{
  vec.reserve(size);
  for (int i = 0; i < size; i++)
  {
    std::string var_i = var + "[" + std::to_string(i) + "]";
    symbol_ptr xi = std::make_shared<symbol>(var_i);
    vec.push_back(xi);
  }
}

// Get the derivative of a function with respect to a vector
matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec)
{
  const size_t len = vec.size();
  matrix jacobian(len, 1);
  for (int i = 0; i < len; i++)
  {
    jacobian(i,0) = func.diff(*vec[i]);
  }
  return jacobian;
}

// Get the derivative of an expression vector with respect to a symbol vector
matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec)
{
  const size_t ex_vec_size = ex_vec.rows();
  const size_t sym_vec_size = sym_vec.size();

  matrix hessian(ex_vec_size, sym_vec_size);
  for (int r = 0; r < ex_vec_size; r++)
  {
    const ex& expr = ex_vec[r];
    for (int c = 0; c < sym_vec_size; c++)
    {
      const symbol_ptr sym = sym_vec[c];
      hessian(r,c) = expr.diff(*sym);
    }
  }
  return hessian;
}