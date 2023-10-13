#ifndef ORLQP_DOUBLE_INTEGRATOR_MPC_MODEL_HPP_
#define ORLQP_DOUBLE_INTEGRATOR_MPC_MODEL_HPP_

#include "QPTypes.hpp"
#include "MPCModel.hpp"

namespace boylan
{
    struct DoubleIntegratorMPCSolution : public MPCSolution
    {
        EigenVector v_star;
    };

    class DoubleIntegratorMPCModel : public MPCModel
    {
    public:
        DoubleIntegratorMPCModel(const Float time_horizon = 1.0, const Float mass = 1.0)
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
            MPCModel::setup();
        }

        DoubleIntegratorMPCSolution MPCtoDoubleIntegratorSolution(const MPCSolution &mpc_solution);

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

    private:
        Float time_horizon_;
        Float mass_;
    };
}

#endif