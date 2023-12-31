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

#define NUM_OBJECTIVES 3

// MPC Params
#define NUM_NODES 1
#define HORIZON_TIME 1.0F
#define DELTA_TIME HORIZON_TIME/NUM_NODES

// Solver Params
#define NUM_STATES 3*(NUM_NODES + 1)
#define NUM_CONSTRAINTS 4*(NUM_NODES + 1)
#define NUM_COEFFICIENTS 0

#define NUM_VARIABLES NUM_STATES+2*NUM_CONSTRAINTS

using namespace GiNaC;

using symbol_ptr = std::shared_ptr<symbol>;

// f(x) = cost function
ex objective_function(std::vector<symbol_ptr>& x, std::vector<symbol_ptr>& coeffs, const int index)
{
  (void) coeffs; // unused
  (void) index; // unused

  ex objective;
  for (int i = 0; i < NUM_STATES; i++)
  {
    objective += (*x[i])*(*x[i]);
  }
  return objective;
}

// g(x) <= 0
matrix inequality_constraints(std::vector<symbol_ptr>& x, std::vector<symbol_ptr>& coeffs, const int index)
{
  (void) coeffs; // unused
  (void) index; // unused

  matrix inequality(NUM_CONSTRAINTS, 1);

  // Initial state based on random gaussian float
  const float mu = 0.0; // mean
  const float sigma = 5.0; // st dev

  auto const seed = std::random_device{}();
  auto urbg = std::mt19937{seed};
  auto norm = std::normal_distribution<float>{mu, sigma};

  const float startX = norm(urbg);
  const float startV = norm(urbg);

  // Boundary constraints
  inequality(0,0) = (*x[0]) - startX;
  inequality(1,0) =  startX - (*x[0]);
  inequality(2,0) = (*x[NUM_NODES]) - startV;
  inequality(3,0) = startV - (*x[NUM_NODES]);

  // Dynamic constraints
  for (int i = 1; i < NUM_NODES; i++)
  {
    const int xi = i;
    const int vi = NUM_NODES + i;
    const int ui = 2*NUM_NODES + i;
    const int ci = 4*i;
    inequality(ci, 0) = (*x[xi]) - (*x[xi-1]) - (*x[vi-1])*DELTA_TIME;
    inequality(ci+1, 0) = -(*x[xi]) + (*x[xi-1]) + (*x[vi-1])*DELTA_TIME;
    inequality(ci+2, 0) = (*x[vi]) - (*x[vi-1]) - (*x[ui-1])*DELTA_TIME;
    inequality(ci+3, 0) = -(*x[vi]) + (*x[vi-1]) + (*x[ui-1])*DELTA_TIME;
  }
  return inequality;
}

std::vector<float> initial_variables(const int index)
{
  std::vector<float> inits(NUM_VARIABLES);
  for (int i = 0; i < NUM_VARIABLES; i++)
  {
    inits[i] = 1.0F;
  }
  return inits;
}

/*



*/

void create_index_vector(std::vector<symbol_ptr>& vec, const int size, const std::string var);
matrix diff_function_by_vec(const ex& func, const std::vector<symbol_ptr>& vec);
matrix diff_vec_by_vec(const matrix& ex_vec, const std::vector<symbol_ptr>& sym_vec);

int main()
{

  srand(time(NULL));

  std::ofstream cu;
  cu.open("/app/GENERATED_LOOKUP.cu");
  cu.clear();

  cu << generate_HEADER(NUM_OBJECTIVES, NUM_VARIABLES, NUM_COEFFICIENTS);

  for (int np = 0; np < NUM_OBJECTIVES; np++)
  {
    // Get initial values
    const std::vector<float> w0 = initial_variables(np);
    printf("Retrieved initial variables.\n");

    // Generate in file
    for (int nv = 0; nv < NUM_VARIABLES; nv++)
    {
      cu << generate_WI_P_N(w0[nv], np, nv);
    }

    // Create expression vectors
    std::vector<symbol_ptr> w;
    std::vector<symbol_ptr> coeffs;
    create_index_vector(w, NUM_VARIABLES, "w");
    create_index_vector(coeffs, NUM_COEFFICIENTS, "c");

    // Get ineq. constraint vector
    const matrix inequality = inequality_constraints(w, coeffs, np);
    printf("Retrieved constraint matrix.\n");

    assert(inequality.rows() == NUM_CONSTRAINTS);

    // Objective function
    ex objective = objective_function(w, coeffs, np);
    printf("Retrieved objective function.\n");

    // Generate in file
    cu << generate_COST_P(objective, np);

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

    assert(KKT.rows() == NUM_VARIABLES);

    // Generate in file
    for (int nv = 0; nv < NUM_VARIABLES; nv++)
    {
      cu << generate_KKT_P_N(KKT(nv,0), np, nv);
    }

    // Get the jacobian of the objective w.r.t the symbols
    const matrix J = diff_vec_by_vec(KKT, w);
    printf("Calculated jacobian.\n");

    // Generate in file
    for (int row = 0; row < NUM_VARIABLES; row++)
    {
      for (int col = 0; col < NUM_VARIABLES; col++)
      {
        cu << generate_J_P_N_M(J(row,col), np, row, col);
      }
    }

  }

  cu << generate_WI_LOOKUP(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_WI_EVALUATE();
  cu << generate_COST_LOOKUP(NUM_OBJECTIVES);
  cu << generate_KKT_LOOKUP(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_KKT_EVALUATE();
  cu << generate_KKT_FORMAT();
  cu << generate_J_LOOKUP(NUM_OBJECTIVES, NUM_VARIABLES);
  cu << generate_J_EVALUATE();
  cu << generate_J_FORMAT();
  cu << generate_ENDER();
  printf("Generated GENERATED_LOOKUP.cu file.\n");

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