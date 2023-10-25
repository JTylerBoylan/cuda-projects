#ifndef ORLQP_QP_PROBLEM_HPP_
#define ORLQP_QP_PROBLEM_HPP_

#include "orlqp/types.hpp"

namespace orlqp
{

    struct QPProblemUpdate
    {
        bool hessian = false,
             gradient = false,
             linear_constraint = false,
             lower_bound = false,
             upper_bound = false;
    };

    struct QPProblem
    {
        /*
            QP Problem:
            minimize 0.5 * x' * Q * x + c' * x
            subject to lb <= Ac * x <= ub

            Q : Hessian [nxn]
            c : Gradient [nx1]
            Ac : Linear constraint matrix [mxn]
            lb : Lower bound [mx1]
            ub : Upper bound [mx1]
        */

        using Ptr = std::shared_ptr<QPProblem>;

        const int num_variables;
        const int num_constraints;
        EigenSparseMatrix hessian;
        EigenVector gradient;
        EigenSparseMatrix linear_constraint;
        EigenVector lower_bound;
        EigenVector upper_bound;

        QPProblemUpdate update;

        QPProblem(const int n, const int m)
            : num_variables(n), num_constraints(m)
        {
            hessian.resize(n, n);
            gradient.resize(n);
            linear_constraint.resize(m, n);
            lower_bound.resize(m);
            upper_bound.resize(m);
        }
    };

    struct QPSolution
    {
        using Ptr = std::shared_ptr<QPSolution>;

        float run_time_s;
        float setup_time_s;
        float solve_time_s;
        EigenVector xstar;
    };
}

#endif