#ifndef ORLQP_QP_PROBLEM_ARRAY_HPP_
#define ORLQP_QP_PROBLEM_ARRAY_HPP_

#include "orlqp/types.hpp"
#include "orlqp/qp_problem.hpp"

namespace orlqp
{

    struct QPProblemArray : public QPProblem
    {

        using Ptr = std::shared_ptr<QPProblemArray>;

        const int num_problems;
        std::vector<int> variable_idx_map;
        std::vector<int> constraint_idx_map;

        QPProblemArray(const int n, const int m, const int Np)
            : QPProblem(n, m), num_problems(Np)
        {
        }
    };

    struct QPSolutionArray
    {
        using Ptr = std::shared_ptr<QPSolutionArray>;

        float run_time_s;
        float setup_time_s;
        float solve_time_s;
        std::vector<EigenVector> xstar;
    };

}

#endif