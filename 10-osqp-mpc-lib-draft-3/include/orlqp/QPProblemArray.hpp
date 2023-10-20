#ifndef ORLQP_QP_PROBLEM_ARRAY_HPP_
#define ORLQP_QP_PROBLEM_ARRAY_HPP_

#include "orlqp/types.hpp"
#include "orlqp/QPProblem.hpp"

namespace orlqp
{

    struct QPArraySolution
    {
        using Ptr = std::shared_ptr<QPArraySolution>;

        float run_time_s;
        float setup_time_s;
        float solve_time_s;
        std::vector<EigenVector> xstar;
    };

    class QPArrayProblem : public QPProblem
    {

    public:
        using Ptr = std::shared_ptr<QPArrayProblem>;

        const int num_problems;
        std::vector<int> variable_idx_map;
        std::vector<int> constraint_idx_map;

        QPArrayProblem(const std::vector<QPProblem::Ptr> qp_array);

        QPArraySolution::Ptr getQPArraySolution();

    private:
        void calculateQPArrayHessian();

        void calculateQPArrayGradient();

        void calculateQPArrayLinearConstraint();

        void calculateQPArrayLowerBound();

        void calculateQPArrayUpperBound();

        void updateQPArrayHessian();

        void updateQPArrayGradient();

        void updateQPArrayLinearConstraint();

        void updateQPArrayLowerBound();

        void updateQPArrayUpperBound();
    };

}

#endif