#ifndef ORLQP_QP_SOLVER_HPP_
#define ORLQP_QP_SOLVER_HPP_

#include "QPTypes.hpp"
#include "QPModel.hpp"

namespace boylan
{

    class QPSolver
    {
    public:
        virtual bool setup(QPModel &model) = 0;
        virtual bool solve(QPModel &model) = 0;

        virtual void updateHessian(EigenSparseMatrix &hessian) = 0;
        virtual void updateGradient(EigenVector &gradient) = 0;
        virtual void updateLinearConstraint(EigenSparseMatrix &lin_constraint) = 0;
        virtual void updateLowerBound(EigenVector &lower_bound) = 0;
        virtual void updateUpperBound(EigenVector &upper_bound) = 0;

        QPSolution &getSolution()
        {
            return qp_solution_;
        }

    protected:
        QPSolution qp_solution_;
    };
}

#endif
