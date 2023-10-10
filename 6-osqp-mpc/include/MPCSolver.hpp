#ifndef ONR_OSQP_MPC_SOLVER_HPP_
#define ONR_OSQP_MPC_SOLVER_HPP_

#include "types.hpp"
#include "MPCDynamics.hpp"
#include "MPCConstraints.hpp"
#include "MPCObjective.hpp"

#include "osqp/osqp.h"

namespace boylan
{

    class MPCSolver
    {
    public:
        using Ptr = std::shared_ptr<MPCSolver>;

        MPCSolver(const OSQPInt n, const OSQPInt m, MPCObjective::Ptr objective, MPCDynamics::Ptr dynamics, MPCConstraints::Ptr constraints)
            : n_(n), m_(m), objective_(objective), dynamics_(dynamics), constraints_(constraints)
        {
            setupSettings();
            setupSolver();
        }

        OSQPSettingsPtr getSettings()
        {
            return settings_;
        }

        OSQPSolver *getSolver()
        {
            return solver_;
        }

        OSQPInt solve()
        {
            return osqp_solve(solver_);
        }

        OSQPSolution *getSolution()
        {
            return solver_->solution;
        }

        ~MPCSolver()
        {
            osqp_cleanup(solver_);
            freeCSC(P_);
            freeCSC(A_);
        }

    private:
        const OSQPInt n_, m_;
        const MPCObjective::Ptr objective_;
        const MPCDynamics::Ptr dynamics_;
        const MPCConstraints::Ptr constraints_;

        OSQPSolver *solver_;
        OSQPSettingsPtr settings_;

        OSQPCscMatrixPtr P_, A_;

        void setupSettings()
        {
            settings_ = std::make_shared<OSQPSettings>();
            osqp_set_default_settings(settings_.get());
        }

        void setupSolver()
        {
            P_ = toCSC(objective_->getHessian());
            float *q = objective_->getGradient().data();
            A_ = toCSC(dynamics_->getLinearConstraintMatrix());
            float *l = constraints_->getLowerBounds().data();
            float *u = constraints_->getUpperBounds().data();
            osqp_setup(&solver_, P_.get(), q, A_.get(), l, u, m_, n_, settings_.get());
        }

        OSQPCscMatrixPtr toCSC(const Eigen::SparseMatrix<OSQPFloat> &eigen_matrix)
        {
            OSQPInt A_nnz = eigen_matrix.nonZeros();
            OSQPFloat *A_x = new OSQPFloat[A_nnz];
            OSQPInt *A_i = new OSQPInt[A_nnz];
            OSQPInt *A_p = new OSQPInt[eigen_matrix.cols() + 1];

            int k = 0;
            A_p[0] = 0;
            for (int j = 0; j < eigen_matrix.outerSize(); ++j)
            {
                for (Eigen::SparseMatrix<float>::InnerIterator it(eigen_matrix, j); it; ++it)
                {
                    A_x[k] = it.value();
                    A_i[k] = it.row();
                    ++k;
                }
                A_p[j + 1] = k;
            }
            OSQPCscMatrixPtr osqp_matrix = std::make_shared<OSQPCscMatrix>();
            csc_set_data(osqp_matrix.get(), eigen_matrix.rows(), eigen_matrix.cols(), A_nnz, A_x, A_i, A_p);
            return osqp_matrix;
        }

        void freeCSC(OSQPCscMatrixPtr osqp_matrix)
        {
            delete osqp_matrix->x;
            delete osqp_matrix->i;
            delete osqp_matrix->p;
        }
    };

}

#endif