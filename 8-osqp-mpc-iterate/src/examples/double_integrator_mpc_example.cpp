#include "DoubleIntegratorMPCModel.hpp"
#include "OSQPSolver.hpp"
#include "MPCProblem.hpp"

#include <iostream>

using namespace boylan;

int main()
{

    EigenVector initial_state(2), desired_state(2);
    initial_state << 2.0, 0.0;
    desired_state << -1.0, 0.0;

    MPCProblem<DoubleIntegratorMPCModel, OSQP> problem;

    problem.updateInitialState(initial_state);
    problem.updateDesiredState(desired_state);

    MPCSolution &mpc_solution = problem.getSolution();
    DoubleIntegratorMPCSolution solution = problem.getModel()->MPCtoDoubleIntegratorSolution(mpc_solution);

    std::cout << "X star:\n" << solution.x_star << std::endl;
    std::cout << "V star:\n" << solution.v_star << std::endl;
    std::cout << "U star:\n" << solution.u_star << std::endl;
    std::cout << "Run time: " << solution.run_time_s << "s" << std::endl;
    std::cout << "Setup time: " << solution.setup_time_s << "s" << std::endl;
    std::cout << "Solve time: " << solution.solve_time_s << "s" << std::endl;

    return 0;
}