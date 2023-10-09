#ifndef ONR_OSQP_MPC_DYNAMICS_HPP_
#define ONR_OSQP_MPC_DYNAMICS_HPP_

#include <assert.h>
#include <Eigen/Sparse>

#include "types.hpp"

namespace boylan
{
    using namespace Eigen;

    class MPCDynamics
    {
    public:
        MPCDynamics(const int N, const int nx, const int nu, const MATRIX &A, const MATRIX &B)
            : N_(N), nx_(nx), nu_(nu), A_(A), B_(B)
        {
            assert(("MPC Dynamics A matrix must be size Nx by Nx.", A_.rows() == nx_ && A_.cols() == nx_));
            assert(("MPC Dynamics B matrix must be size Nx by Nu.", B_.rows() == nx_ && B_.cols() == nu_));
            this->calculateLinearConstraintMatrix();
        }

        SparseMatrix<FLOAT> &getLinearConstraintMatrix()
        {
            return Ac_;
        }

    private:
        const int N_;
        const int nx_, nu_;
        const MATRIX A_, B_;

        SparseMatrix<FLOAT> Ac_;

        void calculateLinearConstraintMatrix()
        {
            Ac_.resize(nx_ * (N_ + 1) + nx_ * (N_ + 1) + nu_ * N_,
                       nx_ * (N_ + 1) + nu_ * N_);

            for (int i = 0; i < nx_ * (N_ + 1); i++)
            {
                Ac_.insert(i, i) = -1;
            }

            for (int i = 0; i < N_; i++)
                for (int j = 0; j < nx_; j++)
                    for (int k = 0; k < nx_; k++)
                    {
                        float value = A_(j, k);
                        if (value != 0)
                        {
                            Ac_.insert(nx_ * (i + 1) + j, nx_ * i + k) = value;
                        }
                    }

            for (int i = 0; i < N_; i++)
                for (int j = 0; j < nx_; j++)
                    for (int k = 0; k < nu_; k++)
                    {
                        float value = B_(j, k);
                        if (value != 0)
                        {
                            Ac_.insert(nx_ * (i + 1) + j, nu_ * i + k + nx_ * (N_ + 1)) = value;
                        }
                    }

            for (int i = 0; i < nx_ * (N_ + 1) + nu_ * N_; i++)
            {
                Ac_.insert(i + (N_ + 1) * nx_, i) = 1;
            }
        }
    };

}

#endif