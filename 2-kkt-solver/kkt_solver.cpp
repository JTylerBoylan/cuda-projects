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

#define NUM_ITERATIONS 20

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

// Set initial values for x, lambda, and s
void initialize(std::vector<float>& x0, std::vector<float>& lambda0, std::vector<float>& s0)
{
  x0 = {-0.9f, -0.1f};
  lambda0 = {1.0f, 1.0f, 1.0f};
  s0 = {1.0f, 1.0f, 1.0f};
}




void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var);

matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec);

matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec);

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
  std::vector<symbol_ptr> x, lambda, s;
  create_index_vector(x, x_len, "x");
  create_index_vector(lambda, lambda_len, "h");
  create_index_vector(s, s_len, "s");

  printf("Expression vectors created.\n");
  std::cout << std::endl;

  // Get ineq. constraint vector
  const matrix g = inequality_constraints(x);

  // Check for valid vector sizes
  const size_t g_size = g.rows();
  if (g_size != lambda_len)
  {
    printf("Incorrect size error: Inequality constraints vector size (%lu) must equal lambda vector size (%lu).\n",
            g_size, lambda_len);
    assert(g_size == lambda_len);
  }

  // Sanity check
  std::cout << "f(x) = " << objective_function(x) << std::endl;
  std::cout << std::endl;

  // Lagrange equation
  ex lagrange_equation = objective_function(x);
  for (int l = 0; l < lambda_len; l++)
  {
    lagrange_equation += (*lambda[l])*(g(l,0) + (*s[l])*(*s[l]));
  }

  // Sanity check
  std::cout << "L(x, h, s) = " << lagrange_equation << std::endl;
  std::cout << std::endl;

  // Create combined symbol vector
  std::vector<symbol_ptr> w;
  w.insert(w.end(), x.begin(), x.end());
  w.insert(w.end(), lambda.begin(), lambda.end());
  w.insert(w.end(), s.begin(), s.end());

  // Get the jacobian of the objective w.r.t the symbols
  const matrix KKT = diff_function_by_vec(lagrange_equation, w);

  // Sanity check
  std::cout << "KKT(x,h,s) = " << std::endl; 
  for (int k = 0; k < KKT.rows(); k++)
  {
    std::cout << "  [";
    std::cout << KKT[k];
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;

  // Get the hessian of the objective w.r.t the symbols
  const matrix hessian = diff_vec_by_vec(KKT, w);

  // Sanity check
  std::cout << "H(x,h,s) = " << std::endl; 
  for (int r = 0; r < hessian.rows(); r++)
  {
    std::cout << "  [\t";
    for (int c = 0; c < hessian.cols(); c++)
    {
      std::cout << hessian(r,c) << "\t";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << std::endl;

  // Get the inverse of the hessian
  const matrix inv_hessian = inverse(hessian);

  // Sanity check
  std::cout << "H^-1 = [" << inv_hessian.rows() << "x" << inv_hessian.cols() << "]" << std::endl;
  std::cout << "KKT = [" << KKT.rows() << "x" << KKT.cols() <<  "]" << std::endl;

  // Get the inverse hessian * KKT
  matrix inv_hess_KKT = inv_hessian.mul(KKT);

  // Sanity check
  std::cout << "H^-1*KKT = [" << inv_hess_KKT.rows() << "x" << inv_hess_KKT.cols() <<  "]" << std::endl;
  std::cout << std::endl;

  // Create combined value vector
  std::vector<float> wi;
  wi.insert(wi.end(), x0.begin(), x0.end());
  wi.insert(wi.end(), lambda0.begin(), lambda0.end());
  wi.insert(wi.end(), s0.begin(), s0.end());

  // Newton-Raphson
  for (int iter = 0; iter < NUM_ITERATIONS; iter++)
  {
    // Create substiution map
    exmap val_map;
    for (int idx = 0; idx < w.size(); idx++)
    {
      val_map[*(w[idx])] = wi[idx];
    }

    // Evaluate function
    ex eval = inv_hess_KKT.subs(val_map);

    // Apply newton raphson
    for (int idx = 0; idx < wi.size(); idx++)
    {
      wi[idx] -= ex_to<numeric>(eval[idx]).to_double();
    }
  }

  std::cout << "Finished solving." << std::endl;
  std::cout << std::endl;

  exmap xmap, hmap, smap;

  // Solution
  std::cout << std::endl;
  std::cout << "Final values: " << std::endl;
  int xi, hi, si;
  for (xi = 0; xi < x.size(); xi++)
  {
    std::cout << "x" << xi << " = " << wi[xi] << std::endl;
    xmap[*x[xi]] = wi[xi];
  }
  std::cout << std::endl;
  for (hi = 0; hi < lambda.size(); hi++)
  {
    std::cout << "lam" << hi << " = " << wi[hi + xi] << std::endl;
    hmap[*lambda[hi]] = wi[hi + xi];
  }
  std::cout << std::endl;
  for (si = 0; si < s.size(); si++)
  {
    std::cout << "s" << si << " = " << wi[si + hi + xi] << std::endl;
    smap[*s[si]] = wi[si + hi + xi];
  }
  std::cout << std::endl;

  std::cout << "Final cost: " << objective_function(x).subs(xmap) << std::endl;
  std::cout << std::endl;


  return 0;
}

// Fill an expression vector with indexed variables
void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var)
{
  vec.reserve(size);
  for (int i = 0; i < size; i++)
  {
    std::string var_i = var + std::to_string(i);
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