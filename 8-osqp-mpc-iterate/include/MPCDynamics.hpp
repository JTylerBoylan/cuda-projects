#ifndef ORL_OSQP_MPC_DYNAMICS_HPP_
#define ORL_OSQP_MPC_DYNAMICS_HPP_

#include <assert.h>

#include "types.hpp"
#include "Eigen/Sparse"

namespace boylan
{
    using namespace Eigen;

    class MPCDynamics
    {
    public:
        using Ptr = std::shared_ptr<MPCDynamics>;

        MPCDynamics(const int N_problems, const int N, const int nx, const int nu)
            : n_probs_(N_problems), N_(N), nx_(nx), nu_(nu)
        {
            const int sizeX = n_probs_ * (nx_ * (N_ + 1) + nx_ * (N_ + 1) + nu_ * N_);
            const int sizeY = n_probs_ * (nx_ * (N_ + 1) + nu_ * N_);
            Ac_.resize(sizeX, sizeY);
        }

        void setDynamics(const int N_problem, const EigenMatrix &A, const EigenMatrix &B)
        {
            assert(("Problem number out of expected range", N_problem < n_probs_));
            assert(("MPC Dynamics A matrix must be size Nx by Nx.", A.rows() == nx_ && A.cols() == nx_));
            assert(("MPC Dynamics B matrix must be size Nx by Nu.", B.rows() == nx_ && B.cols() == nu_));
            this->setLinearConstraintMatrix(N_problem, A, B);
        }

        SparseMatrix<OSQPFloat> &getLinearConstraintMatrix()
        {
            Ac_.setFromTriplets(AC_triplets_.begin(), AC_triplets_.end());
            return Ac_;
        }

    private:
        const int n_probs_;
        const int N_;
        const int nx_, nu_;

        std::vector<Eigen::Triplet<OSQPFloat>> AC_triplets_;
        SparseMatrix<OSQPFloat, ColMajor> Ac_;

        void setLinearConstraintMatrix(const int np, const EigenMatrix &A, const EigenMatrix &B)
        {

            const int startIdxX = np * (nx_ * (N_ + 1) + nx_ * (N_ + 1) + nu_ * N_);
            const int startIdxY = np * (nx_ * (N_ + 1) + nu_ * N_);

            for (int i = 0; i < nx_ * (N_ + 1); i++)
            {
                AC_triplets_.push_back(Eigen::Triplet<OSQPFloat>(startIdxX + i, startIdxY + i, -1));
            }

            for (int i = 0; i < N_; i++)
                for (int j = 0; j < nx_; j++)
                    for (int k = 0; k < nx_; k++)
                    {
                        float value = A(j, k);
                        if (value != 0)
                        {
                            AC_triplets_.push_back(Eigen::Triplet<OSQPFloat>(startIdxX + nx_ * (i + 1) + j, startIdxY + nx_ * i + k, value));
                        }
                    }

            for (int i = 0; i < N_; i++)
                for (int j = 0; j < nx_; j++)
                    for (int k = 0; k < nu_; k++)
                    {
                        float value = B(j, k);
                        if (value != 0)
                        {
                            AC_triplets_.push_back(Eigen::Triplet<OSQPFloat>(startIdxX + nx_ * (i + 1) + j, startIdxY + nu_ * i + k + nx_ * (N_ + 1), value));
                        }
                    }

            for (int i = 0; i < nx_ * (N_ + 1) + nu_ * N_; i++)
            {
                AC_triplets_.push_back(Eigen::Triplet<OSQPFloat>(startIdxX + i + (N_ + 1) * nx_, startIdxY + i, 1));
            }

        }

    };

}

#endif