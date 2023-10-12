#ifndef ORLQP_QP_SOLVER_HPP_
#define ORLQP_QP_SOLVER_HPP_

#include "QPTypes.hpp"
#include "QPProblem.hpp"

namespace boylan
{

    class QPSolver
    {
    public:
        virtual bool solve(QPProblem &problem) = 0;

        virtual bool setup(QPProblem &problem) = 0;

        QPSolution &getSolution()
        {
            return latest_solution_;
        }

    protected:

        QPSolution latest_solution_;
    };
}

#endif
