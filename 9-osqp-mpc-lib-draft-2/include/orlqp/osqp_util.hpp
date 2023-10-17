#ifndef ORLQP_OSQP_UTIL_HPP_
#define ORLQP_OSQP_UTIL_HPP_

#include <execution>

#include "orlqp/types.hpp"
#include "orlqp/osqp_solver.hpp"
#include "orlqp/qp_problem.hpp"

namespace orlqp
{

    OSQP::Ptr qp2osqp(const QPProblem::Ptr qp);

    void to_csc(const EigenSparseMatrix &matrix,
                OSQPCscMatrix *&M, OSQPInt &Mnnz, OSQPFloat *&Mx, OSQPInt *&Mi, OSQPInt *&Mp);

    OSQPInt setup_osqp(OSQP::Ptr osqp);

    OSQPInt solve_osqp(OSQP::Ptr osqp);

    OSQPInt update_settings(OSQP::Ptr osqp);

    void update_data(OSQP::Ptr osqp, QPProblem::Ptr qp);

    QPSolution::Ptr get_solution(OSQP::Ptr osqp);

    template <auto ExecutionPolicy = std::execution::par>
    void solve_multi_osqp(std::vector<OSQP::Ptr> osqps);

}

#endif