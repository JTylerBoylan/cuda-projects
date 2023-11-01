#ifndef TIANZE_LIP_HPP_
#define TIANZE_LIP_HPP_

#include "orlqp/orlqp.hpp"

namespace tianze_lip
{
    using namespace orlqp;

    class TianzeLIP
    {
    public:
        using Ptr = std::shared_ptr<TianzeLIP>;

        const int num_states = 4;
        const int num_controls = 2;
        const int num_slack = 3;

        const int num_steps;
        const int nodes_per_step;
        const int num_nodes;
        const int num_variables;

        EigenVector state_objective;
        EigenVector control_objective;
        EigenVector step_objective;
        EigenVector slack_step_objective;
        EigenVector slack_avoid_objective;
        EigenVector slack_input_objective;

        TianzeLIP(const int Ns, const int Nns)
            : num_steps(Ns), nodes_per_step(Nns),
              num_nodes(Ns * Nns),
              num_variables(num_nodes * (num_states + num_controls + num_slack) + num_steps)
        {
        }

        void setStepReference(const EigenVector &sx_ref, const EigenVector &sy_ref);
        // set u_ref constraints

        void setMaxFootReach(const EigenVector &r_max);
        void setObstaclePositions(const EigenMatrix &p_obs);
        void setInitialPosition(const EigenVector &p_0);
        void setSafeDistance(const Float r_obs);
        void setControlOffset(const EigenVector &u_off);

    private:
    };
}

#endif