#ifndef ORLQP_DOUBLE_INTEGRATOR_MPC_ARRAY_MODEL_HPP_
#define ORLQP_DOUBLE_INTEGRATOR_MPC_ARRAY_MODEL_HPP_

#include "QPTypes.hpp"
#include "mpc/MPCArrayModel.hpp"

namespace boylan
{
    struct DoubleIntegratorMPCArraySolution : public MPCArraySolution
    {
        std::vector<EigenVector> v_star;
    };

    class DoubleIntegratorMPCArrayModel : public MPCArrayModel
    {
    public:
        DoubleIntegratorMPCArrayModel(const size_t num_problems = 0, const Float time_horizon = 1.0, const Float mass = 1.0)
            : MPCArrayModel(num_problems), time_horizon_(time_horizon), mass_(mass)
        {
        }

        void setup(const int id) override
        {
            num_nodes_.resize(num_problems_);
            num_states_.resize(num_problems_);
            num_controls_.resize(num_problems_);
            initial_state_.resize(num_problems_);
            desired_state_.resize(num_problems_);
            state_objective_.resize(num_problems_);
            control_objective_.resize(num_problems_);
            state_dynamics_.resize(num_problems_);
            control_dynamics_.resize(num_problems_);
            state_lower_bound_.resize(num_problems_);
            state_upper_bound_.resize(num_problems_);
            control_lower_bound_.resize(num_problems_);
            control_upper_bound_.resize(num_problems_);
            for (int p = 0; p < this->num_problems_; p++)
            {
                countNodes(p);
                countStates(p);
                countControls(p);
                calculateInitialState(p);
                calculateDesiredState(p);
                calculateStateObjective(p);
                calculateControlObjective(p);
                calculateStateDynamics(p);
                calculateControlDynamics(p);
                calculateStateBounds(p);
                calculateControlBounds(p);
            }
            MPCArrayModel::setup(id);
        }

        DoubleIntegratorMPCArraySolution MPCtoDoubleIntegratorSolution(const MPCArraySolution &mpc_solution);

        void countNodes(const int index);
        void countStates(const int index);
        void countControls(const int index);
        void calculateInitialState(const int index);
        void calculateDesiredState(const int index);
        void calculateStateObjective(const int index);
        void calculateControlObjective(const int index);
        void calculateStateDynamics(const int index);
        void calculateControlDynamics(const int index);
        void calculateStateBounds(const int index);
        void calculateControlBounds(const int index);

    private:
        Float time_horizon_;
        Float mass_;
    };
}

#endif