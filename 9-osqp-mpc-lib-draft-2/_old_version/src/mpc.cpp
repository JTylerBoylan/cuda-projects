#include "mpc/MPCModel.hpp"

namespace boylan
{

    void MPCModel::countVariables()
    {
        const size_t N = getNodeCount();
        const size_t Nx = getStateSize();
        const size_t Nu = getControlSize();
        this->num_variables_ = (Nx * (N + 1) + Nu * N);
    }

    void MPCModel::countConstraints()
    {
        const size_t N = getNodeCount();
        const size_t Nx = getStateSize();
        const size_t Nu = getControlSize();
        this->num_constraints_ = (2 * Nx * (N + 1) + Nu * N);
    }

    void MPCModel::calculateHessianMatrix()
    {
        hessian_triplets_.clear();
        for (int i = 0; i < this->num_nodes_ + 1; i++)
        {
            const int idx = this->num_states_ * i;
            for (int j = 0; j < this->num_states_; j++)
                for (int k = 0; k < this->num_states_; k++)
                {
                    const Float value = this->state_objective_(j, k);
                    if (value != 0)
                        hessian_triplets_.push_back(EigenTriplet(idx + k, idx + k, value));
                }
        }
        for (int i = 0; i < this->num_nodes_; i++)
        {
            const int idx = this->num_states_ * (this->num_nodes_ + 1) + this->num_controls_ * i;
            for (int j = 0; j < this->num_controls_; j++)
                for (int k = 0; k < this->num_controls_; k++)
                {
                    const Float value = this->control_objective_(j, k);
                    if (value != 0)
                        hessian_triplets_.push_back(EigenTriplet(idx + k, idx + k, value));
                }
        }

        hessian_ = EigenSparseMatrix(num_variables_, num_variables_);
        hessian_.setFromTriplets(hessian_triplets_.begin(), hessian_triplets_.end());
    }

    void MPCModel::calculateGradientVector()
    {
        gradient_ = EigenVector(num_variables_);
        const EigenVector grad_x = state_objective_ * (-desired_state_);
        const int Nnx = num_states_ * (num_nodes_ + 1);
        for (int i = 0; i < Nnx; i++)
        {
            int gIdx = i % num_states_;
            gradient_(i, 0) = grad_x(gIdx, 0);
        }
        for (int j = 0; j < num_controls_ * num_nodes_; j++)
        {
            gradient_(Nnx + j, 0) = 0;
        }
    }

    void MPCModel::calculateLinearConstraintMatrix()
    {
        lin_constraint_triplets_.clear();
        for (int i = 0; i < num_states_ * (num_nodes_ + 1); i++)
            lin_constraint_triplets_.push_back(EigenTriplet(i, i, -1));

        for (int i = 0; i < num_nodes_; i++)
            for (int j = 0; j < num_states_; j++)
                for (int k = 0; k < num_states_; k++)
                {
                    const Float value = state_dynamics_(j, k);
                    if (value != 0)
                        lin_constraint_triplets_.push_back(EigenTriplet(num_states_ * (i + 1) + j, num_states_ * i + k, value));
                }

        for (int i = 0; i < num_nodes_; i++)
            for (int j = 0; j < num_states_; j++)
                for (int k = 0; k < num_controls_; k++)
                {
                    const Float value = control_dynamics_(j, k);
                    if (value != 0)
                        lin_constraint_triplets_.push_back(EigenTriplet(
                            num_states_ * (i + 1) + j,
                            num_controls_ * i + k + num_states_ * (num_nodes_ + 1),
                            value));
                }

        for (int i = 0; i < num_variables_; i++)
            lin_constraint_triplets_.push_back(EigenTriplet(i + (num_nodes_ + 1) * num_states_, i, 1));

        lin_constraint_ = EigenSparseMatrix(num_constraints_, num_variables_);
        lin_constraint_.setFromTriplets(lin_constraint_triplets_.begin(), lin_constraint_triplets_.end());
    }

    void MPCModel::calculateBoundVectors()
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

        lower_bound_ = EigenVector(num_constraints_);
        lower_bound_.block(0, 0, num_eq, 1) = lower_equality;
        lower_bound_.block(num_eq, 0, num_ineq, 1) = lower_inequality;

        upper_bound_ = EigenVector(num_constraints_);
        upper_bound_.block(0, 0, num_eq, 1) = upper_equality;
        upper_bound_.block(num_eq, 0, num_ineq, 1) = upper_inequality;
    }

}