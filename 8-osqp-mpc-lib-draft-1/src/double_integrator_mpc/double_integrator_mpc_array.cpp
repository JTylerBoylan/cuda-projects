#include "double_integrator_mpc/DoubleIntegratorMPCArrayModel.hpp"

#define DI_MPC_NUMBER_OF_NODES 11

#define DI_MPC_NUMBER_OF_STATES 2
#define DI_MPC_NUMBER_OF_CONTROLS 1

#define DI_MPC_INITIAL_POSITION 1.0F
#define DI_MPC_INITIAL_VELOCITY 0.0F

#define DI_MPC_DESIRED_POSITION 0.0F
#define DI_MPC_DESIRED_VELOCITY 0.0F

#define DI_MPC_POSITION_ERROR_COST_WEIGHT 1.0
#define DI_MPC_VELOCITY_ERROR_COST_WEIGHT 0.0
#define DI_MPC_FORCE_COST_WEIGHT 0.0

#define DI_MPC_MIN_POSITION -10.0
#define DI_MPC_MAX_POSITION +10.0
#define DI_MPC_MIN_VELOCITY -10.0
#define DI_MPC_MAX_VELOCITY +10.0
#define DI_MPC_MIN_FORCE -5.0
#define DI_MPC_MAX_FORCE +5.0

namespace boylan
{

    DoubleIntegratorMPCArraySolution DoubleIntegratorMPCArrayModel::MPCtoDoubleIntegratorSolution(const MPCArraySolution &mpc_solution)
    {
        /* TODO */
        return DoubleIntegratorMPCArraySolution();
    }

    void DoubleIntegratorMPCArrayModel::countNodes(const int index)
    {
        this->num_nodes_[index] = DI_MPC_NUMBER_OF_NODES;
    }

    void DoubleIntegratorMPCArrayModel::countStates(const int index)
    {
        this->num_states_[index] = DI_MPC_NUMBER_OF_STATES;
    }

    void DoubleIntegratorMPCArrayModel::countControls(const int index)
    {
        this->num_controls_[index] = DI_MPC_NUMBER_OF_CONTROLS;
    }

    void DoubleIntegratorMPCArrayModel::calculateInitialState(const int index)
    {
        this->initial_state_[index] = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->initial_state_[index] << DI_MPC_INITIAL_POSITION, DI_MPC_INITIAL_VELOCITY;
    }

    void DoubleIntegratorMPCArrayModel::calculateDesiredState(const int index)
    {
        this->desired_state_[index] = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->desired_state_[index] << DI_MPC_DESIRED_POSITION, DI_MPC_DESIRED_VELOCITY;
    }

    void DoubleIntegratorMPCArrayModel::calculateStateObjective(const int index)
    {
        this->state_objective_[index] = EigenMatrix(DI_MPC_NUMBER_OF_STATES, DI_MPC_NUMBER_OF_STATES);
        this->state_objective_[index] << DI_MPC_POSITION_ERROR_COST_WEIGHT, 0.0, 0.0, DI_MPC_VELOCITY_ERROR_COST_WEIGHT;
    }

    void DoubleIntegratorMPCArrayModel::calculateControlObjective(const int index)
    {
        this->control_objective_[index] = EigenMatrix(DI_MPC_NUMBER_OF_CONTROLS, DI_MPC_NUMBER_OF_CONTROLS);
        this->control_objective_[index] << DI_MPC_FORCE_COST_WEIGHT;
    }

    void DoubleIntegratorMPCArrayModel::calculateStateDynamics(const int index)
    {
        const Float delta_time = time_horizon_ / this->num_nodes_[index];
        this->state_dynamics_[index] = EigenMatrix(DI_MPC_NUMBER_OF_STATES, DI_MPC_NUMBER_OF_STATES);
        this->state_dynamics_[index] << 1.0, delta_time, 0.0, 1.0;
    }

    void DoubleIntegratorMPCArrayModel::calculateControlDynamics(const int index)
    {
        const Float delta_time = time_horizon_ / this->num_nodes_[index];
        this->control_dynamics_[index] = EigenMatrix(DI_MPC_NUMBER_OF_STATES, DI_MPC_NUMBER_OF_CONTROLS);
        this->control_dynamics_[index] << 0.0, delta_time / mass_;
    }

    void DoubleIntegratorMPCArrayModel::calculateStateBounds(const int index)
    {
        this->state_lower_bound_[index] = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->state_upper_bound_[index] = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->state_lower_bound_[index] << DI_MPC_MIN_POSITION, DI_MPC_MIN_VELOCITY;
        this->state_upper_bound_[index] << DI_MPC_MAX_POSITION, DI_MPC_MAX_VELOCITY;
    }

    void DoubleIntegratorMPCArrayModel::calculateControlBounds(const int index)
    {
        this->control_lower_bound_[index] = EigenVector(DI_MPC_NUMBER_OF_CONTROLS);
        this->control_upper_bound_[index] = EigenVector(DI_MPC_NUMBER_OF_CONTROLS);
        this->control_lower_bound_[index] << DI_MPC_MIN_FORCE;
        this->control_upper_bound_[index] << DI_MPC_MAX_FORCE;
    }

}