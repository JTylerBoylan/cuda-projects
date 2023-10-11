#ifndef ORL_OSQP_MPC_OBJECTIVE_HPP_
#define ORL_OSQP_MPC_OBJECTIVE_HPP_

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

        MPCObjective(const int N_problems, const int N, const int nx, const int nu)
            : n_probs_(N_problems), N_(N), nx_(nx), nu_(nu)
        {
            const int size = n_probs_ * (nx_ * (N_ + 1) + nu_ * N_);
            gradient_ = EigenVector::Zero(size, 1);
            hessian_.resize(size, size);
        }

        void setObjective(const int N_problem, const EigenMatrix &Q, const EigenMatrix &R, const EigenVector &xf)
        {
            assert(("Problem number out of expected range", N_problem < n_probs_));
            assert(("MPC Objective Q matrix must be size Nx by Nx.", Q.rows() == nx_ && Q.cols() == nx_));
            assert(("MPC Objective R matrix must be size Nu by Nu.", R.rows() == nu_ && R.cols() == nu_));
            assert(("MPC Objective xf vector must be size Nx by 1.", xf.rows() == nx_));
            this->setGradient(N_problem, Q, xf);
            this->setHessian(N_problem, Q, R);
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
        const int n_probs_;
        const int N_;
        const int nx_, nu_;

        EigenVector gradient_;
        SparseMatrix<OSQPFloat, ColMajor> hessian_;

        void setGradient(const int i, const EigenMatrix &Q, const EigenVector &xf)
        {
            const int startIdx = i * (nx_ * (N_ + 1) + nu_ * N_);

            EigenMatrix Qx_ref(nx_, 1);
            Qx_ref = Q * (-xf);

            // populate the gradient vector
            for (int i = 0; i < nx_ * (N_ + 1); i++)
            {
                int posQ = i % nx_;
                float value = Qx_ref(posQ, 0);
                gradient_(startIdx + i, 0) = value;
            }
        }

        void setHessian(const int np, const EigenMatrix &Q, const EigenMatrix &R)
        {

            const int startIdx = np * (nx_ * (N_ + 1) + nu_ * N_);

            // populate hessian matrix
            for (int i = 0; i < nx_ * (N_ + 1) + nu_ * N_; i++)
            {
                if (i < nx_ * (N_ + 1))
                {
                    int posQ = i % nx_;
                    float value = Q.diagonal()[posQ];
                    if (value != 0)
                        hessian_.insert(startIdx + i, startIdx + i) = value;
                }
                else
                {
                    int posR = i % nu_;
                    float value = R.diagonal()[posR];
                    if (value != 0)
                        hessian_.insert(startIdx + i, startIdx + i) = value;
                }
            }
        }
    };

}

#endif