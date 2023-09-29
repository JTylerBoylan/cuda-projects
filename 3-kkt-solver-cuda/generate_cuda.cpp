#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <math.h>
#include <vector>
#include <random>
#include <time.h>
#include <memory>
#include <assert.h>
#include <ginac/ginac.h>

#include "generate_util.cpp"

#define NUM_OBJECTIVES 100
#define NUM_STATES 2
#define NUM_CONSTRAINTS 3
#define NUM_COEFFICIENTS 2

using namespace GiNaC;

using symbol_ptr = std::shared_ptr<symbol>;

// f(x) = cost function
ex objective_function(std::vector<symbol_ptr>& x, std::vector<symbol_ptr>& coeffs, const int index)
{
  return (*coeffs[0])*(*x[0])*(*x[0]) + (*coeffs[1])*(*x[1])*(*x[1]);
}

// g(x) <= 0
matrix inequality_constraints(std::vector<symbol_ptr>& x, std::vector<symbol_ptr>& coeffs, const int index)
{
  (void) coeffs; // unused
  matrix constraints(NUM_CONSTRAINTS,1);
  constraints(0,0) = (*x[0]) - 1;
  constraints(1,0) = (*x[1]) - 0;
  constraints(2,0) = -(*x[0]) + 1;
  return constraints;
}

std::vector<float> initial_variables(const int index)
{
  const float max_val = 1.0;
  const float min_val = -1.0;
  return {
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // x0
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // x1
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // lam0
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // lam1
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // lam2
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // s0
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // s1
    (max_val - min_val)*(((float) rand()) / ((float) RAND_MAX)) + min_val, // s2
  };
}

/*



*/

#define NUM_VARIABLES NUM_STATES+2*NUM_CONSTRAINTS

void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var);
matrix calculate_inv_J_KKT(std::vector<symbol_ptr> w, const ex& objective, const matrix& inequality);

int main()
{

  srand(time(NULL));

  std::ofstream cu;
  cu.open("/app/GENERATED_LOOKUP.cu");
  cu.clear();

  cu << generate_lookup_header(NUM_OBJECTIVES, NUM_VARIABLES, NUM_COEFFICIENTS);

  for (int obj = 0; obj < NUM_OBJECTIVES; obj++)
  {
    // Get initial values
    const std::vector<float> w0 = initial_variables(obj);
    printf("Retrieved initial variables.\n");

    // Create expression vectors
    std::vector<symbol_ptr> w;
    std::vector<symbol_ptr> coeffs;
    create_index_vector(w, NUM_VARIABLES, "w");
    create_index_vector(coeffs, NUM_COEFFICIENTS, "c");

    // Get ineq. constraint vector
    const matrix inequality = inequality_constraints(w, coeffs, obj);
    printf("Retrieved constraint matrix.\n");

    assert(inequality.rows() == NUM_CONSTRAINTS);

    // Objective function
    ex objective = objective_function(w, coeffs, obj);
    printf("Retrieved objective function.\n");

    matrix invjkkt = calculate_inv_J_KKT(w, objective, inequality);

    cu << generate_cost_function(objective, obj);

    for (int v = 0; v < NUM_VARIABLES; v++)
    {
      cu << generate_winitial_definition(w0[v], obj, v);
      cu << generate_expression_function(invjkkt(v,0), obj, v);
    }
  }

  cu << generate_lookup_intercept(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_lookup_initials(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_lookup_objective(NUM_OBJECTIVES);
  cu << generate_lookup_ender();
  printf("Generated _GENERATED_LOOKUP.cu file.\n");

  printf("Done.\n");

  return 0;
}

/*

*/

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

matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec);
matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec);

// Generate inv(J)*KKT given the objective and constraints
matrix calculate_inv_J_KKT(std::vector<symbol_ptr> w, const ex& objective, const matrix& inequality)
{
  // Lagrange equation
  ex lagrange_equation = objective;
  for (int l = 0; l < NUM_CONSTRAINTS; l++)
  {
    const symbol_ptr lambda = w[NUM_STATES+l];
    const symbol_ptr s = w[NUM_STATES+NUM_CONSTRAINTS+l];
    lagrange_equation += (*lambda)*(inequality(l,0) + (*s)*(*s));
  }
  printf("Calculated lagrangian equation.\n");

  // Get the jacobian of the objective w.r.t the symbols
  const matrix KKT = diff_function_by_vec(lagrange_equation, w);
  printf("Calculated KKT conditions.\n");

  // Get the jacobian of the objective w.r.t the symbols
  const matrix J = diff_vec_by_vec(KKT, w);
  printf("Calculated jacobian.\n");

  // Get the inverse of the jacobian
  const matrix inv_J = inverse(J);
  printf("Calculated inverse jacobian.\n");

  // Get the inverse jacobian * KKT
  matrix inv_J_KKT = inv_J.mul(KKT);
  printf("Calculated inverse jacobian by KKT conditions.\n");

  return inv_J_KKT;
}

// Get the derivative of a function with respect to a vector
matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec)
{
  const size_t len = vec.size();
  matrix J(len, 1);
  for (int i = 0; i < len; i++)
  {
    J(i,0) = func.diff(*vec[i]);
  }
  return J;
}

// Get the derivative of an expression vector with respect to a symbol vector
matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec)
{
  const size_t ex_vec_size = ex_vec.rows();
  const size_t sym_vec_size = sym_vec.size();

  matrix J(ex_vec_size, sym_vec_size);
  for (int r = 0; r < ex_vec_size; r++)
  {
    const ex& expr = ex_vec[r];
    for (int c = 0; c < sym_vec_size; c++)
    {
      const symbol_ptr sym = sym_vec[c];
      J(r,c) = expr.diff(*sym);
    }
  }
  return J;
}