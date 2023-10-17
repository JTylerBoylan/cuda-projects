#ifndef ORLQP_OSQP_UTIL_HPP_
#define ORLQP_OSQP_UTIL_HPP_

#include "orlqp/types.hpp"
#include "orlqp/osqp_solver.hpp"
#include "orlqp/qp_problem.hpp"

namespace orlqp
{

    OSQP::Ptr qp2osqp(const QPProblem::Ptr qp);

    void to_csc(const EigenSparseMatrix &matrix,
                OSQPCscMatrix *&M, OSQPInt &Mnnz, OSQPFloat *&Mx, OSQPInt *&Mi, OSQPInt *&Mp);

    OSQPInt solve_osqp(OSQP::Ptr osqp);

    OSQPInt update_settings(OSQP::Ptr osqp);

}

#endif