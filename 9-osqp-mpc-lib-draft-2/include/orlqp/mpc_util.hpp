#ifndef ORLQP_MPC_UTIL_HPP_
#define ORLQP_MPC_UTIL_HPP_

#include "orlqp/types.hpp"
#include "orlqp/qp_problem.hpp"
#include "orlqp/mpc_problem.hpp"

namespace orlqp
{

    MPCProblem::Ptr create_mpc(const int Nx, const int Nu, const int Nn);

    QPProblem::Ptr mpc2qp(const MPCProblem::Ptr mpc);

    MPCSolution::Ptr get_mpc_solution(const int Nx, const int Nu, const int Nn,
                                      const QPSolution::Ptr qp_solution);

    void to_hessian(const int n,
                    EigenSparseMatrix &H,
                    const int Nx, const int Nu, const int Nn,
                    const EigenMatrix &Q, const EigenMatrix &R);

    void to_gradient(const int n,
                     EigenVector &G,
                     const int Nx, const int Nu, const int Nn,
                     const EigenMatrix &Q, const EigenVector xf);

    void to_linear_constraint(const int n, const int m,
                              EigenSparseMatrix &Ac,
                              const int Nx, const int Nu, const int Nn,
                              const EigenMatrix &A, const EigenMatrix &B);

    void to_lower_bound(const int m,
                        EigenVector &lb,
                        const int Nx, const int Nu, const int Nn,
                        const EigenVector x0,
                        const EigenVector &x_min, const EigenVector &x_max,
                        const EigenVector &u_min, const EigenVector &u_max);

    void to_upper_bound(const int m,
                        EigenVector &ub,
                        const int Nx, const int Nu, const int Nn,
                        const EigenVector x0,
                        const EigenVector &x_min, const EigenVector &x_max,
                        const EigenVector &u_min, const EigenVector &u_max);

    void update_initial_state(QPProblem::Ptr qp, const int Nx, const EigenVector &x0);

}

#endif