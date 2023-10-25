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

        virtual int updateHessian(EigenSparseMatrix &hessian) = 0;
        virtual int updateGradient(EigenVector &gradient) = 0;
        virtual int updateLinearConstraint(EigenSparseMatrix &lin_constraint) = 0;
        virtual int updateLowerBound(EigenVector &lower_bound) = 0;
        virtual int updateUpperBound(EigenVector &upper_bound) = 0;

        QPSolution &getSolution()
        {
            return qp_solution_;
        }

    protected:
        QPSolution qp_solution_;
    };
}

#endif
