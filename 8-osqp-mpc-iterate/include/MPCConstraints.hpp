#ifndef ORL_OSQP_MPC_CONSTRAINTS_HPP_
#define ORL_OSQP_MPC_CONSTRAINTS_HPP_

#include <assert.h>

#include "types.hpp"

namespace boylan
{

    using namespace Eigen;

    class MPCConstraints
    {

    public:
        using Ptr = std::shared_ptr<MPCConstraints>;

        MPCConstraints(const int N_problems, const int N, const int nx, const int nu)
            : n_probs_(N_problems), N_(N), nx_(nx), nu_(nu)
        {
            const int sizeEq = nx_ * (N_ + 1);
            const int sizeIneq = nx_ * (N_ + 1) + nu_ * N_;
            const int size = n_probs_ * (sizeEq + sizeIneq);
            lower_bounds_ = EigenMatrix::Zero(size, 1);
            upper_bounds_ = EigenMatrix::Zero(size, 1);
        }

        void setConstraints(const int N_problem, const EigenVector &x0,
                            const EigenVector &x_min, const EigenVector &x_max,
                            const EigenVector &u_min, const EigenVector &u_max)

        {
            assert(("Problem number out of expected range", N_problem < n_probs_));
            assert(("MPC Constraints X vectors must be size Nx by 1.",
                    x0.rows() == nx_ && x_min.rows() == nx_ && x_max.rows() == nx_));
            assert(("MPC Constraints U vectors must be size Nu by 1.",
                    u_min.rows() == nu_ && u_max.rows() == nu_));
            this->setLowerAndUpperBounds(N_problem, x0, x_min, x_max, u_min, u_max);
        }

        EigenVector &getLowerBounds()
        {
            return lower_bounds_;
        }

        EigenVector &getUpperBounds()
        {
            return upper_bounds_;
        }

        void updateX0(const int n_problem, const EigenVector &x0)
        {
            assert(("MPC Constraints X vectors must be size Nx by 1.", x0.rows() == nx_));
            const int sizeEq = nx_ * (N_ + 1);
            const int sizeIneq = nx_ * (N_ + 1) + nu_ * N_;
            const int startIdx = n_problem * (sizeEq + sizeIneq);
            lower_bounds_.block(startIdx, 0, nx_, 1) = -x0;
            upper_bounds_.block(startIdx, 0, nx_, 1) = -x0;
        }

    private:
        const int n_probs_;
        const int N_;
        const int nx_, nu_;

        EigenVector lower_bounds_, upper_bounds_;

        void setLowerAndUpperBounds(const int np, const EigenVector &x0,
                                    const EigenVector &x_min, const EigenVector &x_max,
                                    const EigenVector &u_min, const EigenVector &u_max)
        {
            const int sizeEq = nx_ * (N_ + 1);
            const int sizeIneq = nx_ * (N_ + 1) + nu_ * N_;
            const int startIdx = np * (sizeEq + sizeIneq);

            EigenVector lower_inequality = EigenMatrix::Zero(sizeIneq, 1);
            EigenVector upper_inequality = EigenMatrix::Zero(sizeIneq, 1);
            for (int i = 0; i < N_ + 1; i++)
            {
                lower_inequality.block(nx_ * i, 0, nx_, 1) = x_min;
                upper_inequality.block(nx_ * i, 0, nx_, 1) = x_max;
            }
            for (int i = 0; i < N_; i++)
            {
                lower_inequality.block(nu_ * i + nx_ * (N_ + 1), 0, nu_, 1) = u_min;
                upper_inequality.block(nu_ * i + nx_ * (N_ + 1), 0, nu_, 1) = u_max;
            }

            EigenVector lower_equality = EigenMatrix::Zero(sizeEq, 1);
            EigenVector upper_equality;
            lower_equality.block(0, 0, nx_, 1) = -x0;
            upper_equality = lower_equality;
            lower_equality = lower_equality;

            lower_bounds_.block(startIdx, 0, sizeEq, 1) = lower_equality;
            lower_bounds_.block(startIdx + sizeEq, 0, sizeIneq, 1) = lower_inequality;

            upper_bounds_.block(startIdx, 0, sizeEq, 1) = upper_equality;
            upper_bounds_.block(startIdx + sizeEq, 0, sizeIneq, 1) = upper_inequality;
        }
    };

}

#endif