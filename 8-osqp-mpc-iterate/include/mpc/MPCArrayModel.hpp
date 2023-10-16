#ifndef ORLQP_MPC_ARRAY_MODEL_HPP_
#define ORLQP_MPC_ARRAY_MODEL_HPP_

#include "QPTypes.hpp"
#include "QPModel.hpp"

namespace boylan
{

    struct MPCArraySolution : public QPSolution
    {
        std::vector<EigenVector> x_star;
        std::vector<EigenVector> u_star;
    };

    class MPCArrayModel : public QPModel
    {
    public:
        /*
            MPC Problem:
            minimize x'*Q*x + u'*R*u
            subject to x(k+1) = A*x(k) + B*u(k)
                       x_min <= x <= x_max
                       u_min <= u <= u_max

            Q : State objective
            R : Control objective
            A : State dynamics
            B : Control dynamics
        */

        MPCArrayModel()
        {
        }

        MPCArrayModel(const size_t num_problems)
            : num_problems_(num_problems)
        {
        }

        size_t &getVariableStartIndex(const int index)
        {
            return var_idx_[index];
        }

        size_t &getConstraintStartIndex(const int index)
        {
            return con_idx_[index];
        }

        size_t &getNodeCount(const int index)
        {
            return num_nodes_[index];
        }

        size_t &getStateSize(const int index)
        {
            return num_states_[index];
        }

        size_t &getControlSize(const int index)
        {
            return num_controls_[index];
        }

        EigenVector &getInitialState(const int index)
        {
            return initial_state_[index];
        }

        EigenVector &getDesiredState(const int index)
        {
            return desired_state_[index];
        }

        EigenMatrix &getStateObjective(const int index)
        {
            return state_objective_[index];
        }

        EigenMatrix &getControlObjective(const int index)
        {
            return control_objective_[index];
        }

        EigenMatrix &getStateDynamics(const int index)
        {
            return state_dynamics_[index];
        }

        EigenMatrix &getControlDynamics(const int index)
        {
            return control_dynamics_[index];
        }

        EigenVector &getStateLowerBound(const int index)
        {
            return state_lower_bound_[index];
        }

        EigenVector &getStateUpperBound(const int index)
        {
            return state_upper_bound_[index];
        }

        EigenVector &getControlLowerBound(const int index)
        {
            return control_lower_bound_[index];
        }

        EigenVector &getControlUpperBound(const int index)
        {
            return control_upper_bound_[index];
        }

        EigenMatrix getSingleHessianMatrix(const int index);
        EigenVector getSingleGradientVector(const int index);
        EigenMatrix getSingleLinearConstraintMatrix(const int index);
        EigenVector getSingleLowerBoundVector(const int index);
        EigenVector getSingleUpperBoundVector(const int index);

        void setup(const int id) override
        {
            (void)id; // unused
            countVariables();
            countConstraints();
            calculateHessianMatrix();
            calculateGradientVector();
            calculateLinearConstraintMatrix();
            calculateBoundVectors();
        }

        void countVariables();
        void countConstraints();
        void calculateHessianMatrix();
        void calculateGradientVector();
        void calculateLinearConstraintMatrix();
        void calculateBoundVectors();

    protected:
        size_t num_problems_;

        std::vector<size_t> num_nodes_;
        std::vector<size_t> num_states_;
        std::vector<size_t> num_controls_;

        std::vector<size_t> var_idx_;
        std::vector<size_t> con_idx_;

        std::vector<EigenVector> initial_state_;
        std::vector<EigenVector> desired_state_;
        std::vector<EigenMatrix> state_objective_;
        std::vector<EigenMatrix> control_objective_;
        std::vector<EigenMatrix> state_dynamics_;
        std::vector<EigenMatrix> control_dynamics_;
        std::vector<EigenVector> state_lower_bound_;
        std::vector<EigenVector> state_upper_bound_;
        std::vector<EigenVector> control_lower_bound_;
        std::vector<EigenVector> control_upper_bound_;
    };

}

#endif