#include "mpc/MPCArrayModel.hpp"

namespace boylan
{

    void MPCArrayModel::countVariables()
    {
        this->var_idx_.resize(num_problems_);
        size_t num_vars = 0;
        for (int p = 0; p < num_problems_; p++)
        {
            this->var_idx_[p] = num_vars;
            const size_t N = getNodeCount(p);
            const size_t Nx = getStateSize(p);
            const size_t Nu = getControlSize(p);
            num_vars += (Nx * (N + 1) + Nu * N);
        }
        this->num_variables_ = num_vars;
    }

    void MPCArrayModel::countConstraints()
    {
        this->con_idx_.resize(num_problems_);
        size_t num_cons = 0;
        for (int p = 0; p < num_problems_; p++)
        {
            this->con_idx_[p] = num_cons;
            const size_t N = getNodeCount(p);
            const size_t Nx = getStateSize(p);
            const size_t Nu = getControlSize(p);
            num_cons += (2 * Nx * (N + 1) + Nu * N);
        }
        this->num_constraints_ = num_cons;
    }

    void MPCArrayModel::calculateHessianMatrix()
    {
        hessian_ = EigenSparseMatrix(num_variables_, num_variables_);
        hessian_triplets_.clear();
        for (int p = 0; p < num_problems_; p++)
        {
            const size_t start_idx = var_idx_[p];
            const size_t N = this->num_nodes_[p];
            const size_t Nx = this->num_states_[p];
            const size_t Nu = this->num_controls_[p];
            for (int i = 0; i < N + 1; i++)
            {
                const int idx = start_idx + Nx * i;
                for (int j = 0; j < Nx; j++)
                    for (int k = 0; k < Nx; k++)
                    {
                        const Float value = this->state_objective_[p](j, k);
                        if (value != 0)
                            hessian_triplets_.push_back(EigenTriplet(idx + k, idx + k, value));
                    }
            }
            for (int i = 0; i < N; i++)
            {
                const int idx = start_idx + Nx * (N + 1) + Nu * i;
                for (int j = 0; j < Nu; j++)
                    for (int k = 0; k < Nu; k++)
                    {
                        const Float value = this->control_objective_[p](j, k);
                        if (value != 0)
                            hessian_triplets_.push_back(EigenTriplet(idx + k, idx + k, value));
                    }
            }
        }
        hessian_.setFromTriplets(hessian_triplets_.begin(), hessian_triplets_.end());
    }

    void MPCArrayModel::calculateGradientVector()
    {
        gradient_ = EigenVector(num_variables_);
        for (int p = 0; p < num_problems_; p++)
        {
            const size_t start_idx = var_idx_[p];
            const size_t N = this->num_nodes_[p];
            const size_t Nx = this->num_states_[p];
            const size_t Nu = this->num_controls_[p];
            const EigenVector grad_x = state_objective_[p] * (-desired_state_[p]);
            const int NNx = Nx * (N + 1);
            for (int i = 0; i < NNx; i++)
            {
                int gIdx = i % Nx;
                gradient_(start_idx + i, 0) = grad_x(gIdx, 0);
            }
            for (int j = 0; j < Nu * N; j++)
            {
                gradient_(start_idx + NNx + j, 0) = 0;
            }
        }
    }

    void MPCArrayModel::calculateLinearConstraintMatrix()
    {
        lin_constraint_ = EigenSparseMatrix(num_constraints_, num_variables_);
        lin_constraint_triplets_.clear();
        for (int p = 0; p < num_problems_; p++)
        {
            size_t start_ridx = con_idx_[p];
            size_t start_cidx = var_idx_[p];

            const size_t N = this->num_nodes_[p];
            const size_t Nx = this->num_states_[p];
            const size_t Nu = this->num_controls_[p];

            for (int i = 0; i < Nx * (N + 1); i++)
                lin_constraint_triplets_.push_back(EigenTriplet(start_ridx + i, start_cidx + i, -1));

            for (int i = 0; i < N; i++)
                for (int j = 0; j < Nx; j++)
                    for (int k = 0; k < Nx; k++)
                    {
                        const Float value = state_dynamics_[p](j, k);
                        if (value != 0)
                            lin_constraint_triplets_.push_back(EigenTriplet(
                                start_ridx + Nx * (i + 1) + j,
                                start_cidx + Nx * i + k,
                                value));
                    }

            for (int i = 0; i < N; i++)
                for (int j = 0; j < Nx; j++)
                    for (int k = 0; k < Nu; k++)
                    {
                        const Float value = control_dynamics_[p](j, k);
                        if (value != 0)
                            lin_constraint_triplets_.push_back(EigenTriplet(
                                start_ridx + Nx * (i + 1) + j,
                                start_cidx + Nx * (N + 1) + Nu * i + k,
                                value));
                    }

            for (int i = 0; i < Nx * (N + 1) + Nu * N; i++)
                lin_constraint_triplets_.push_back(EigenTriplet(start_ridx + Nx * (N + 1) + i, start_cidx + i, 1));
        }
        lin_constraint_.setFromTriplets(lin_constraint_triplets_.begin(), lin_constraint_triplets_.end());
    }

    void MPCArrayModel::calculateBoundVectors()
    {
        lower_bound_ = EigenVector(num_constraints_);
        upper_bound_ = EigenVector(num_constraints_);

        for (int p = 0; p < num_problems_; p++)
        {
            size_t start_idx = con_idx_[p];
            const size_t N = this->num_nodes_[p];
            const size_t Nx = this->num_states_[p];
            const size_t Nu = this->num_controls_[p];

            const int num_eq = Nx * (N + 1);
            const int num_ineq = Nx * (N + 1) + Nu * N;

            EigenVector lower_inequality = EigenVector::Zero(num_ineq);
            EigenVector upper_inequality = EigenVector::Zero(num_ineq);
            for (int i = 0; i < N + 1; i++)
            {
                lower_inequality.block(Nx * i, 0, Nx, 1) = state_lower_bound_[p];
                upper_inequality.block(Nx * i, 0, Nx, 1) = state_upper_bound_[p];
            }
            for (int i = 0; i < N; i++)
            {
                lower_inequality.block(Nu * i + Nx * (N + 1), 0, Nu, 1) = control_lower_bound_[p];
                upper_inequality.block(Nu * i + Nx * (N + 1), 0, Nu, 1) = control_upper_bound_[p];
            }

            EigenVector lower_equality = EigenVector::Zero(num_eq);
            lower_equality.block(0, 0, Nx, 1) = -initial_state_[p];
            EigenVector upper_equality = lower_equality;

            lower_bound_.block(start_idx, 0, num_eq, 1) = lower_equality;
            lower_bound_.block(start_idx + num_eq, 0, num_ineq, 1) = lower_inequality;

            upper_bound_.block(start_idx, 0, num_eq, 1) = upper_equality;
            upper_bound_.block(start_idx + num_eq, 0, num_ineq, 1) = upper_inequality;
        }
    }

    EigenMatrix MPCArrayModel::getSingleHessianMatrix(const int index)
    {
        const int N = num_nodes_[index];
        const int Nx = num_states_[index];
        const int Nu = num_controls_[index];
        const int var_idx = var_idx_[index];
        const int num_vars = Nx * (N + 1) + Nu * N;
        auto hessian = hessian_.block(var_idx, var_idx, num_vars, num_vars);
        return hessian;
    }

    EigenVector MPCArrayModel::getSingleGradientVector(const int index)
    {
        const int N = num_nodes_[index];
        const int Nx = num_states_[index];
        const int Nu = num_controls_[index];
        const int var_idx = var_idx_[index];
        const int num_vars = Nx * (N + 1) + Nu * N;
        auto gradient = gradient_.block(var_idx, 0, num_vars, 1);
        return gradient;
    }

    EigenMatrix MPCArrayModel::getSingleLinearConstraintMatrix(const int index)
    {
        const int N = num_nodes_[index];
        const int Nx = num_states_[index];
        const int Nu = num_controls_[index];
        const int var_idx = var_idx_[index];
        const int con_idx = con_idx_[index];
        const int num_vars = Nx * (N + 1) + Nu * N;
        const int num_cons = 2 * Nx * (N + 1) + Nu * N;
        auto lin_constraint = lin_constraint_.block(con_idx, var_idx, num_cons, num_vars);
        return lin_constraint;
    }

    EigenVector MPCArrayModel::getSingleLowerBoundVector(const int index)
    {
        const int N = num_nodes_[index];
        const int Nx = num_states_[index];
        const int Nu = num_controls_[index];
        const int con_idx = con_idx_[index];
        const int num_cons = 2 * Nx * (N + 1) + Nu * N;
        auto lower_bound = lower_bound_.block(con_idx, 0, num_cons, 1);
        return lower_bound;
    }

    EigenVector MPCArrayModel::getSingleUpperBoundVector(const int index)
    {
        const int N = num_nodes_[index];
        const int Nx = num_states_[index];
        const int Nu = num_controls_[index];
        const int con_idx = con_idx_[index];
        const int num_cons = 2 * Nx * (N + 1) + Nu * N;
        auto upper_bound = upper_bound_.block(con_idx, 0, num_cons, 1);
        return upper_bound;
    }

}