#ifndef ORLQP_OSQP_SOLVER_HPP_
#define ORLQP_OSQP_SOLVER_HPP_

#include "QPTypes.hpp"
#include "QPSolver.hpp"

#include "osqp/osqp.h"

namespace boylan
{

    class OSQP : public QPSolver
    {
    public:
        OSQP();

        bool solve(QPModel &model) override;
        bool setup(QPModel &model) override;

        void updateHessian(EigenSparseMatrix &hessian);
        void updateGradient(EigenVector &gradient);
        void updateLinearConstraint(EigenSparseMatrix &lin_constraint);
        void updateLowerBound(EigenVector &lower_bound);
        void updateUpperBound(EigenVector &upper_bound);

        int getLatestExit()
        {
            return latest_exit_;
        }

        std::shared_ptr<OSQPSettings> getSettings()
        {
            return osqp_settings_;
        }

        ~OSQP();

    private:
        int latest_exit_ = -1;

        std::shared_ptr<OSQPSolver *> osqp_solver_;
        std::shared_ptr<OSQPSettings> osqp_settings_;
        std::shared_ptr<OSQPSolution *> osqp_solution_;

        std::shared_ptr<OSQPCscMatrix> hessian_csc_;
        std::shared_ptr<OSQPCscMatrix> lin_constraint_csc_;

        std::shared_ptr<OSQPCscMatrix> convertSparseMatrixToCSC(const EigenSparseMatrix &matrix);

        void freeCSC(std::shared_ptr<OSQPCscMatrix> matrix);
    };

}

#endif