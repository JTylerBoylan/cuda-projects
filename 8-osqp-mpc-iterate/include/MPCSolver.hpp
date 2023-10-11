#ifndef ORL_OSQP_MPC_SOLVER_HPP_
#define ORL_OSQP_MPC_SOLVER_HPP_

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
            settings_ = new OSQPSettings();
            osqp_set_default_settings(settings_);
        }

        OSQPSettings *getSettings()
        {
            return settings_;
        }

        OSQPSolver *getSolver()
        {
            return solver_;
        }

        OSQPSolution *getSolution()
        {
            return solver_->solution;
        }

        OSQPInt setup()
        {
            P_ = toCSC(objective_->getHessian());
            OSQPFloat *q = objective_->getGradient().data();
            dynamics_->getLinearConstraintMatrix().makeCompressed();
            A_ = toCSC(dynamics_->getLinearConstraintMatrix());
            OSQPFloat *l = constraints_->getLowerBounds().data();
            OSQPFloat *u = constraints_->getUpperBounds().data();
            return osqp_setup(&solver_, P_, q, A_, l, u, m_, n_, settings_);
        }

        OSQPInt solve()
        {
            return osqp_solve(solver_);
        }

        void updateX0(const int n_problem, const EigenVector& x0)
        {
            constraints_->updateX0(n_problem, x0);
            osqp_update_data_vec(solver_, nullptr, constraints_->getLowerBounds().data(), constraints_->getUpperBounds().data());
        }

        ~MPCSolver()
        {
            osqp_cleanup(solver_);
            freeCSC(P_);
            delete P_;
            freeCSC(A_);
            delete A_;
            delete settings_;
        }

    private:
        const OSQPInt n_, m_;
        const MPCObjective::Ptr objective_;
        const MPCDynamics::Ptr dynamics_;
        const MPCConstraints::Ptr constraints_;

        OSQPSolver *solver_;
        OSQPSettings *settings_;

        OSQPCscMatrix *P_, *A_;

        OSQPCscMatrix *toCSC(const Eigen::SparseMatrix<OSQPFloat> &eigen_matrix)
        {
            OSQPInt A_nnz = eigen_matrix.nonZeros();
            OSQPFloat *A_x = new OSQPFloat[A_nnz];
            OSQPInt *A_i = new OSQPInt[A_nnz];
            OSQPInt *A_p = new OSQPInt[eigen_matrix.cols() + 1];

            int k = 0;
            A_p[0] = 0;
            for (int j = 0; j < eigen_matrix.outerSize(); ++j)
            {
                for (Eigen::SparseMatrix<OSQPFloat>::InnerIterator it(eigen_matrix, j); it; ++it)
                {
                    A_x[k] = it.value();
                    A_i[k] = it.row();
                    ++k;
                }
                A_p[j + 1] = k;
            }
            OSQPCscMatrix *osqp_matrix = new OSQPCscMatrix();
            csc_set_data(osqp_matrix, eigen_matrix.rows(), eigen_matrix.cols(), A_nnz, A_x, A_i, A_p);
            return osqp_matrix;
        }

        void freeCSC(OSQPCscMatrix *osqp_matrix)
        {
            delete osqp_matrix->x;
            delete osqp_matrix->i;
            delete osqp_matrix->p;
        }
    };

}

#endif