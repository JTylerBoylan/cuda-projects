#include "DoubleIntegratorMPCProblem.hpp"

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

    DoubleIntegratorMPCSolution DoubleIntegratorMPCProblem::MPCtoDoubleIntegratorSolution(const MPCSolution &mpc_solution)
    {
        DoubleIntegratorMPCSolution dimpc_solution;
        dimpc_solution.x_star = EigenVector(num_nodes_ + 1);
        dimpc_solution.v_star = EigenVector(num_nodes_ + 1);
        for (int i = 0; i < num_nodes_ + 1; i++)
        {
            dimpc_solution.x_star(i) = mpc_solution.x_star(num_states_ * i);
            dimpc_solution.v_star(i) = mpc_solution.x_star(num_states_ * i + 1);
        }
        dimpc_solution.u_star = mpc_solution.u_star;
        dimpc_solution.run_time_s = mpc_solution.run_time_s;
        dimpc_solution.setup_time_s = mpc_solution.setup_time_s;
        dimpc_solution.solve_time_s = mpc_solution.solve_time_s;
        return dimpc_solution;
    }

    DoubleIntegratorMPCSolution DoubleIntegratorMPCProblem::QPtoDoubleIntegratorSolution(const QPSolution &qp_solution)
    {
        return this->MPCtoDoubleIntegratorSolution(this->QPtoMPCSolution(qp_solution));
    }

    void DoubleIntegratorMPCProblem::countNodes()
    {
        this->num_nodes_ = DI_MPC_NUMBER_OF_NODES;
    }

    void DoubleIntegratorMPCProblem::countStates()
    {
        this->num_states_ = DI_MPC_NUMBER_OF_STATES;
    }

    void DoubleIntegratorMPCProblem::countControls()
    {
        this->num_controls_ = DI_MPC_NUMBER_OF_CONTROLS;
    }

    void DoubleIntegratorMPCProblem::calculateInitialState()
    {
        this->initial_state_ = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->initial_state_ << DI_MPC_INITIAL_POSITION, DI_MPC_INITIAL_VELOCITY;
    }

    void DoubleIntegratorMPCProblem::calculateDesiredState()
    {
        this->desired_state_ = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->desired_state_ << DI_MPC_DESIRED_POSITION, DI_MPC_DESIRED_VELOCITY;
    }

    void DoubleIntegratorMPCProblem::calculateStateObjective()
    {
        this->state_objective_ = EigenMatrix(DI_MPC_NUMBER_OF_STATES, DI_MPC_NUMBER_OF_STATES);
        this->state_objective_ << DI_MPC_POSITION_ERROR_COST_WEIGHT, 0.0, 0.0, DI_MPC_VELOCITY_ERROR_COST_WEIGHT;
    }

    void DoubleIntegratorMPCProblem::calculateControlObjective()
    {
        this->control_objective_ = EigenMatrix(DI_MPC_NUMBER_OF_CONTROLS, DI_MPC_NUMBER_OF_CONTROLS);
        this->control_objective_ << DI_MPC_FORCE_COST_WEIGHT;
    }

    void DoubleIntegratorMPCProblem::calculateStateDynamics()
    {
        const Float delta_time = time_horizon_ / this->num_nodes_;
        this->state_dynamics_ = EigenMatrix(DI_MPC_NUMBER_OF_STATES, DI_MPC_NUMBER_OF_STATES);
        this->state_dynamics_ << 1.0, delta_time, 0.0, 1.0;
    }

    void DoubleIntegratorMPCProblem::calculateControlDynamics()
    {
        const Float delta_time = time_horizon_ / this->num_nodes_;
        this->control_dynamics_ = EigenMatrix(DI_MPC_NUMBER_OF_STATES, DI_MPC_NUMBER_OF_CONTROLS);
        this->control_dynamics_ << 0.0, delta_time / mass_;
    }

    void DoubleIntegratorMPCProblem::calculateStateBounds()
    {
        this->state_lower_bound_ = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->state_upper_bound_ = EigenVector(DI_MPC_NUMBER_OF_STATES);
        this->state_lower_bound_ << DI_MPC_MIN_POSITION, DI_MPC_MIN_VELOCITY;
        this->state_upper_bound_ << DI_MPC_MAX_POSITION, DI_MPC_MAX_VELOCITY;
    }

    void DoubleIntegratorMPCProblem::calculateControlBounds()
    {
        this->control_lower_bound_ = EigenVector(DI_MPC_NUMBER_OF_CONTROLS);
        this->control_upper_bound_ = EigenVector(DI_MPC_NUMBER_OF_CONTROLS);
        this->control_lower_bound_ << DI_MPC_MIN_FORCE;
        this->control_upper_bound_ << DI_MPC_MAX_FORCE;
    }

}