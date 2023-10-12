#ifndef ORLQP_DOUBLE_INTEGRATOR_MPC_PROBLEM_HPP_
#define ORLQP_DOUBLE_INTEGRATOR_MPC_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "MPCProblem.hpp"

namespace boylan
{
    struct DoubleIntegratorMPCSolution : public MPCSolution
    {
        EigenVector v_star;
    };

    class DoubleIntegratorMPCProblem : public MPCProblem
    {
    public:
        DoubleIntegratorMPCProblem(const Float time_horizon = 1.0, const Float mass = 1.0)
            : time_horizon_(time_horizon), mass_(mass)
        {
        }

        void setup() override
        {
            countNodes();
            countStates();
            countControls();
            calculateInitialState();
            calculateDesiredState();
            calculateStateObjective();
            calculateControlObjective();
            calculateStateDynamics();
            calculateControlDynamics();
            calculateStateBounds();
            calculateControlBounds();
            MPCProblem::setup();
        }

        DoubleIntegratorMPCSolution MPCtoDoubleIntegratorSolution(const MPCSolution &mpc_solution);

        DoubleIntegratorMPCSolution QPtoDoubleIntegratorSolution(const QPSolution &qp_solution);

    private:
        Float time_horizon_;
        Float mass_;

        void countNodes();
        void countStates();
        void countControls();
        void calculateInitialState();
        void calculateDesiredState();
        void calculateStateObjective();
        void calculateControlObjective();
        void calculateStateDynamics();
        void calculateControlDynamics();
        void calculateStateBounds();
        void calculateControlBounds();
    };
}

#endif