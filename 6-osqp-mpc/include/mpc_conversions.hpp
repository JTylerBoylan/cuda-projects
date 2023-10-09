#ifndef ONR_OSQP_MPC_CONVERSIONS_HPP_
#define ONR_OSQP_MPC_CONVERSIONS_HPP_

#include "mpc_types.hpp"

#include "OsqpEigen/OsqpEigen.h"

namespace boylan
{

    using namespace Eigen;

    MPCResult solveMPC(const MPCProblem& problem)
    {
        /* TODO */
        auto solver = toSolver(problem);
        // Solve
        // Convert results
    }

    OsqpEigen::Solver toSolver(const MPCProblem& problem)
    {
        /* TODO */
        // Convert Q,R to P
        // Convert A,B to A'
        // Convert xm,um to lb, ub
    }

}

#endif