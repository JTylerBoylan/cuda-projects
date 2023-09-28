#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <math.h>
#include <vector>
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
  matrix constraints(NUM_CONSTRAINTS,1);
  constraints(0,0) = (*x[0]) - 1;
  constraints(1,0) = (*x[1]) - 0;
  constraints(2,0) = -(*x[0]) + 1;
  return constraints;
}

std::vector<float> initial_variables()
{
  return {
    -0.9, // x0
    0.1, // x1
    1.0, // lam0
    1.0, // lam1
    1.0, // lam2
    1.0, // s0
    1.0, // s1
    1.0, // s2
  };
}


void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var);

matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec);

matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec);


int main()
{

  // Get initial values
  const std::vector<float> w0 = initial_variables();
  printf("Retrieved initial variables.\n");

  // Create expression vectors
  std::vector<symbol_ptr> w;
  create_index_vector(w, NUM_VARIABLES, "w");

  // Get ineq. constraint vector
  const matrix g = inequality_constraints(w);
  printf("Retrieved constraint matrix.\n");

  assert(g.rows() == NUM_CONSTRAINTS);

  // Objective function
  ex objective = objective_function(w);
  printf("Retrieved objective function.\n");

  // Lagrange equation
  ex lagrange_equation = objective;
  for (int l = 0; l < NUM_CONSTRAINTS; l++)
  {
    const symbol_ptr lambda = w[NUM_STATES+l];
    const symbol_ptr s = w[NUM_STATES+NUM_CONSTRAINTS+l];
    lagrange_equation += (*lambda)*(g(l,0) + (*s)*(*s));
  }
  printf("Calculated lagrangian equation.\n");

  // Get the jacobian of the objective w.r.t the symbols
  const matrix KKT = diff_function_by_vec(lagrange_equation, w);
  printf("Calculated KKT conditions.\n");

  // Get the jacobian of the objective w.r.t the symbols
  const matrix jacobian = diff_vec_by_vec(KKT, w);
  printf("Calculated jacobian.\n");

  // Get the inverse of the jacobian
  const matrix inv_jacobian = inverse(jacobian);
  printf("Calculated inverse jacobian.\n");

  // Get the inverse jacobian * KKT
  matrix inv_jacobian_KKT = inv_jacobian.mul(KKT);
  printf("Calculated inverse jacobian by KKT conditions.\n");

  std::ofstream cu;
  cu.open("/app/GENERATED_LOOKUP.cu");
  cu.clear();
  cu << generate_lookup_header(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_cost_function(objective);
  for (int p = 0; p < NUM_OBJECTIVES; p++)
  {
    for (int v = 0; v < NUM_VARIABLES; v++)
    {
      cu << generate_winitial_definition(w0[v], p, v);
      cu << generate_expression_function(inv_jacobian_KKT(v,0), p, v);
    }
  }
  cu << generate_lookup_intercept(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_lookup_initials(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_lookup_ender();
  printf("Generated _GENERATED_LOOKUP.cu file.\n");

  printf("Done.\n");

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

  matrix jacobian(ex_vec_size, sym_vec_size);
  for (int r = 0; r < ex_vec_size; r++)
  {
    const ex& expr = ex_vec[r];
    for (int c = 0; c < sym_vec_size; c++)
    {
      const symbol_ptr sym = sym_vec[c];
      jacobian(r,c) = expr.diff(*sym);
    }
  }
  return jacobian;
}