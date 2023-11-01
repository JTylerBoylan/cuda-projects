#include <iostream>

#include "orlqp/orlqp.hpp"

using namespace orlqp;
int main(void)
{
    std::cout << "Hello world!\n";

    // Params
    const int lookahead_steps = 2;
    const int nodes_per_step = 4;

    const double stride_frequency = 1.0;
    const double stride_length = 1.0;
    const double delT = 1.0 / (stride_frequency * nodes_per_step);

    const double g = 9.81;
    const double z0 = 1.0;

    // Create MPC
    const int num_states = 4;
    const int num_controls = 2;
    const int num_nodes = lookahead_steps * nodes_per_step;
    MPCProblem::Ptr MPC = std::make_shared<MPCProblem>(num_states, num_controls, num_nodes);

    // Initial state
    MPC->x0 << 0, 0, 0, 0; /* TODO */

    // Desired state
    MPC->xf << 0, 0, 0, 0; /* TODO */

    // State objective
    const double Wx = 1.0;
    const double Wy = 1.0;
    const double Wdx = 1.0;
    const double Wdy = 1.0;
    MPC->state_objective << Wx, 0, 0, 0,
        0, Wy, 0, 0,
        0, 0, Wdx, 0,
        0, 0, 0, Wdy;

    // Control objective
    const double Wcopx = 1.0;
    const double Wcopy = 1.0;
    MPC->control_objective << Wcopx, 0, 0, Wcopy;

    // State dynamics
    MPC->state_dynamics << 1, 0, 1, 0,
        0, 1, 0, 1,
        g * delT / z0, 1, 0, 0,
        0, g * delT / z0, 0, 1;

    // Control dynamics
    MPC->control_dynamics << 0, 0,
        0, 0,
        g * delT / z0, 0,
        0, g * delT / z0;

    // Bounds
    const double inf = std::numeric_limits<double>::infinity();
    MPC->x_min << -inf, -inf, -inf, -inf;
    MPC->x_max << +inf, +inf, +inf, +inf;
    MPC->u_min << -inf, -inf;
    MPC->u_max << +inf, +inf;

    return 0;
}
