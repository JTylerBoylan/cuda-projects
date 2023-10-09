#include "mpc_types.hpp"
#include "mpc_conversions.hpp"

using namespace boylan;

int main()
{

    MPCDynamics dynamics;
    dynamics.A << 0.0;
    dynamics.B << 0.0;

    MPCConstraints constraints;
    constraints.x_min << 0.0;
    constraints.x_max << 0.0;
    constraints.u_min << 0.0;
    constraints.u_max << 0.0;

    MPCObjective objective;
    objective.Q << 0.0;
    objective.R << 0.0;

    const MPCProblem problem{dynamics, constraints, objective};

    MPCResult result = solveMPC(problem);

    return 0;
}