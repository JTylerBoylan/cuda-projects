#ifndef ONR_OSQP_MPC_OBJECTIVE_HPP_
#define ONR_OSQP_MPC_OBJECTIVE_HPP_

#include <assert.h>

#include "types.hpp"
#include "Eigen/Sparse"

namespace boylan
{

    using namespace Eigen;

    class MPCObjective
    {
    public:
        using Ptr = std::shared_ptr<MPCObjective>;

        MPCObjective(const int N, const int nx, const int nu, const EigenMatrix &Q, const EigenMatrix &R, const EigenMatrix xf)
            : N_(N), nx_(nx), nu_(nu), Q_(Q), R_(R), xf_(xf)
        {
            assert(("MPC Objective Q matrix must be size Nx by Nx.", Q_.rows() == nx_ && Q_.cols() == nx_));
            assert(("MPC Objective R matrix must be size Nu by Nu.", R_.rows() == nu_ && R_.cols() == nu_));
            assert(("MPC Objective xf vector must be size Nx by 1.", xf_.rows() == nx_));
            this->calculateGradient();
            this->calculateHessian();
        }

        EigenVector &getGradient()
        {
            return gradient_;
        }

        SparseMatrix<OSQPFloat> &getHessian()
        {
            return hessian_;
        }

    private:
        const int N_;
        const int nx_, nu_;
        const EigenMatrix Q_, R_;
        const EigenVector xf_;

        EigenVector gradient_;
        SparseMatrix<OSQPFloat, ColMajor> hessian_;

        void calculateGradient()
        {
            EigenMatrix Qx_ref(nx_, 1);
            Qx_ref = Q_ * (-xf_);

            // populate the gradient vector
            gradient_ = EigenVector::Zero(nx_ * (N_ + 1) + nu_ * N_, 1);
            for (int i = 0; i < nx_ * (N_ + 1); i++)
            {
                int posQ = i % nx_;
                float value = Qx_ref(posQ, 0);
                gradient_(i, 0) = value;
            }
        }

        void calculateHessian()
        {
            hessian_.resize(nx_ * (N_ + 1) + nu_ * N_,
                            nx_ * (N_ + 1) + nu_ * N_);

            // populate hessian matrix
            for (int i = 0; i < nx_ * (N_ + 1) + nu_ * N_; i++)
            {
                if (i < nx_ * (N_ + 1))
                {
                    int posQ = i % nx_;
                    float value = Q_.diagonal()[posQ];
                    if (value != 0)
                        hessian_.insert(i, i) = value;
                }
                else
                {
                    int posR = i % nu_;
                    float value = R_.diagonal()[posR];
                    if (value != 0)
                        hessian_.insert(i, i) = value;
                }
            }
        }
    };

}

#endif