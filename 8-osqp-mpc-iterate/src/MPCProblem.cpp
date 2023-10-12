#include "MPCProblem.hpp"

namespace boylan
{

    void MPCProblem::countVariables()
    {
        const size_t N = nodeCount();
        const size_t Nx = stateSize();
        const size_t Nu = controlSize();
        this->num_var_ = (2 * Nx * (N + 1) + Nu * N);
    }

    void MPCProblem::countConstraints()
    {
        const size_t N = nodeCount();
        const size_t Nx = stateSize();
        const size_t Nu = controlSize();
        this->num_con_ = (Nx * (N + 1) + Nu * N);
    }

    void MPCProblem::calculateHessianMatrix()
    {
        hessian_triplets_.clear();
        for (int i = 0; i < this->num_con_; i++)
        {
            if (i < num_states_ * (num_nodes_ + 1))
            {
                const int sIdx = i % num_states_;
                const float value = this->state_dynamics_.diagonal()[sIdx];
                if (value != 0)
                    hessian_triplets_.push_back(EigenTriplet(i, i, value));
            }
            else
            {
                const int cIdx = i % num_controls_;
                const float value = this->control_dynamics_.diagonal()[cIdx];
                if (value != 0)
                    hessian_triplets_.push_back(EigenTriplet(i, i, value));
            }
        }
        hessian_.setFromTriplets(hessian_triplets_.begin(), hessian_triplets_.end());
    }

    void MPCProblem::calculateGradientVector()
    {
        gradient_.resize(num_var_, 1);
        const EigenVector grad_x = state_objective_ * (-desired_state_);
        for (int i = 0; i < num_states_ * (num_nodes_ + 1); i++)
        {
            int gIdx = i % num_states_;
            gradient_(i, 0) = grad_x(gIdx, 0);
        }
    }

    void MPCProblem::calculateLinearConstraintMatrix()
    {
        lin_constraint_triplets_.clear();
        for (int i = 0; i < num_states_ * (num_nodes_ + 1); i++)
        {
            lin_constraint_triplets_.push_back(EigenTriplet(i, i, -1));
        }

        for (int i = 0; i < num_nodes_; i++)
            for (int j = 0; j < num_states_; j++)
                for (int k = 0; k < num_states_; k++)
                {
                    const float value = state_dynamics_(j, k);
                    if (value != 0)
                    {
                        lin_constraint_triplets_.push_back(EigenTriplet(num_states_ * (i + 1) + j, num_states_ * i + k, value));
                    }
                }

        for (int i = 0; i < num_nodes_; i++)
            for (int j = 0; j < num_states_; j++)
                for (int k = 0; k < num_controls_; k++)
                {
                    const float value = control_dynamics_(j, k);
                    if (value != 0)
                    {
                        lin_constraint_triplets_.push_back(EigenTriplet(
                            num_states_ * (i + 1) + j,
                            num_controls_ * i + k + num_states_ * (num_nodes_ + 1),
                            value));
                    }
                }

        for (int i = 0; i < num_con_; i++)
        {
            lin_constraint_triplets_.push_back(EigenTriplet(i + (num_nodes_ + 1) * num_states_, i, 1));
        }
        lin_constraint_.setFromTriplets(lin_constraint_triplets_.begin(), lin_constraint_triplets_.end());
    }

    void MPCProblem::calculateBoundVectors()
    {
        const int num_eq = num_states_ * (num_nodes_ + 1);
        const int num_ineq = num_states_ * (num_nodes_ + 1) + num_controls_ * num_nodes_;

        EigenVector lower_inequality = EigenVector::Zero(num_ineq);
        EigenVector upper_inequality = EigenVector::Zero(num_ineq);
        for (int i = 0; i < num_nodes_ + 1; i++)
        {
            lower_inequality.block(num_states_ * i, 0, num_states_, 1) = state_lower_bound_;
            upper_inequality.block(num_states_ * i, 0, num_states_, 1) = state_upper_bound_;
        }
        for (int i = 0; i < num_nodes_; i++)
        {
            lower_inequality.block(num_controls_ * i + num_states_ * (num_nodes_ + 1), 0, num_controls_, 1) = control_lower_bound_;
            upper_inequality.block(num_controls_ * i + num_states_ * (num_nodes_ + 1), 0, num_controls_, 1) = control_upper_bound_;
        }

        EigenVector lower_equality = EigenVector::Zero(num_eq);
        lower_equality.block(0, 0, num_states_, 1) = -initial_state_;
        EigenVector upper_equality = lower_equality;

        lower_bound_.block(0, 0, num_eq, 1) = lower_equality;
        lower_bound_.block(num_eq, 0, num_ineq, 1) = lower_inequality;

        upper_bound_.block(0, 0, num_eq, 1) = upper_equality;
        upper_bound_.block(num_eq, 0, num_ineq, 1) = upper_inequality;
    }

}