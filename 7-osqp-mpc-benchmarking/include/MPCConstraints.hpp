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

        MPCConstraints(const int N, const int nx, const int nu, const EigenVector &x0,
                       const EigenVector &x_min, const EigenVector &x_max,
                       const EigenVector &u_min, const EigenVector &u_max)
            : N_(N), nx_(nx), nu_(nu), x0_(x0), x_min_(x_min), x_max_(x_max), u_min_(u_min), u_max_(u_max)
        {
            assert(("MPC Constraints X vectors must be size Nx by 1.", x_min_.rows() == nx_ && x_max_.rows() == nx_));
            assert(("MPC Constraints U vectors must be size Nu by 1.", u_min_.rows() == nu_ && u_max_.rows() == nu_));
            this->calculateLowerAndUpperBounds();
        }

        EigenVector &getLowerBounds()
        {
            return lower_bounds_;
        }

        EigenVector &getUpperBounds()
        {
            return upper_bounds_;
        }

        void updateX0(const EigenVector &x0)
        {
            x0_ = x0;
            lower_bounds_.block(0, 0, 12, 1) = -x0;
            upper_bounds_.block(0, 0, 12, 1) = -x0;
        }

    private:
        const int N_;
        const int nx_, nu_;
        EigenVector x0_;
        const EigenVector x_min_, x_max_, u_min_, u_max_;

        EigenVector lower_bounds_, upper_bounds_;

        void calculateLowerAndUpperBounds()
        {
            EigenVector lowerInequality = EigenMatrix::Zero(nx_ * (N_ + 1) + nu_ * N_, 1);
            EigenVector upperInequality = EigenMatrix::Zero(nx_ * (N_ + 1) + nu_ * N_, 1);
            for (int i = 0; i < N_ + 1; i++)
            {
                lowerInequality.block(nx_ * i, 0, nx_, 1) = x_min_;
                upperInequality.block(nx_ * i, 0, nx_, 1) = x_max_;
            }
            for (int i = 0; i < N_; i++)
            {
                lowerInequality.block(nu_ * i + nx_ * (N_ + 1), 0, nu_, 1) = u_min_;
                upperInequality.block(nu_ * i + nx_ * (N_ + 1), 0, nu_, 1) = u_max_;
            }

            EigenVector lowerEquality = EigenMatrix::Zero(nx_ * (N_ + 1), 1);
            EigenVector upperEquality;
            lowerEquality.block(0, 0, nx_, 1) = -x0_;
            upperEquality = lowerEquality;
            lowerEquality = lowerEquality;

            lower_bounds_ = EigenMatrix::Zero(2 * nx_ * (N_ + 1) + nu_ * N_, 1);
            lower_bounds_ << lowerEquality, lowerInequality;

            upper_bounds_ = EigenMatrix::Zero(2 * nx_ * (N_ + 1) + nu_ * N_, 1);
            upper_bounds_ << upperEquality, upperInequality;
        }
    };

}

#endif