#ifndef ORLQP_MPC_PROBLEM_HPP_
#define ORLQP_MPC_PROBLEM_HPP_

#include "orlqp/types.hpp"

namespace orlqp
{

    struct MPCProblem
    {
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

        using Ptr = std::shared_ptr<MPCProblem>;

        const int num_states;
        const int num_controls;
        const int num_nodes;
        EigenVector x0, xf;
        EigenMatrix state_objective;
        EigenMatrix control_objective;
        EigenMatrix state_dynamics;
        EigenMatrix control_dynamics;
        EigenVector x_min, x_max;
        EigenVector u_min, u_max;

        MPCProblem(const int Nx, const int Nu, const int N)
            : num_states(Nx), num_controls(Nu), num_nodes(N)
        {
            state_objective.resize(Nx, Nx);
            control_objective.resize(Nu, Nu);
            state_dynamics.resize(Nx, Nx);
            control_dynamics.resize(Nx, Nu);
            x_min.resize(Nx);
            x_max.resize(Nx);
            u_min.resize(Nu);
            u_max.resize(Nu);
        }
    };

    struct MPCSolution
    {
        using Ptr = std::shared_ptr<MPCSolution>;

        float run_time_s;
        float setup_time_s;
        float solve_time_s;
        EigenVector xstar;
        EigenVector ustar;
    };

}

#endif