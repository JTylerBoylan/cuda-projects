#ifndef ORLQP_QP_PROBLEM_HPP_
#define ORLQP_QP_PROBLEM_HPP_

#include "QPTypes.hpp"

namespace boylan
{

    struct QPSolution
    {
        EigenVector x_star;
        Float run_time_s;
        Float setup_time_s;
        Float solve_time_s;
    };

    class QPProblem
    {
    public:
        /*
            QP Problem:
            minimize 0.5 * x' * Q * x + c' * x
            subject to lb <= Ac * x <= ub

            Q : Hessian
            c : Gradient
            Ac : Linear constraint matrix
            lb : Lower bound
            ub : Upper bound
        */

       using Ptr = std::shared_ptr<QPProblem>;

        size_t &variableCount()
        {
            return num_var_;
        }

        size_t &constraintCount()
        {
            return num_con_;
        }

        EigenSparseMatrix &getHessianMatrix()
        {
            return hessian_;
        }

        EigenVector &getGradientVector()
        {
            return gradient_;
        }

        EigenSparseMatrix &getLinearConstraintMatrix()
        {
            return lin_constraint_;
        }

        EigenVector &getLowerBoundVector()
        {
            return lower_bound_;
        }

        EigenVector &getUpperBoundVector()
        {
            return upper_bound_;
        }

        virtual void setup() {}

    protected:
        size_t num_var_;
        size_t num_con_;

        EigenSparseMatrix hessian_;
        EigenVector gradient_;
        EigenSparseMatrix lin_constraint_;
        EigenVector lower_bound_;
        EigenVector upper_bound_;

        std::vector<EigenTriplet> hessian_triplets_;
        std::vector<EigenTriplet> lin_constraint_triplets_;
    };

}

#endif