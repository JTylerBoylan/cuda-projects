#include "double_integrator_mpc/DoubleIntegratorMPCArrayModel.hpp"
#include "osqp/OSQPSolver.hpp"
#include "mpc/MPCArrayProblem.hpp"

#include <iostream>

using namespace boylan;

int main()
{

    EigenVector initial_state(2), desired_state(2);
    initial_state << -4.0, 0.0;
    desired_state << 0.0, 0.0;

    const int num_problems = 1000;
    MPCArrayProblem<DoubleIntegratorMPCArrayModel, OSQP> problem(num_problems);
    problem.setup();

    for (int p = 0; p < num_problems; p++)
    {
        problem.updateInitialState(p, initial_state);
        // problem.updateDesiredState(p, desired_state);
    }

    std::cout << "500 Gradient:\n"
              << problem.getModel()->getSingleGradientVector(500) << std::endl;
    // std::cout << "Gradient:\n" << problem.getModel()->getGradientVector() << std::endl;
    // std::cout << "Lower Bound:\n" << problem.getModel()->getLowerBoundVector() << std::endl;

    problem.solve();

    MPCArraySolution &mpc_solution = problem.getSolution();

    std::cout << "X star:\n"
              << mpc_solution.x_star[500] << std::endl;
    std::cout << "U star:\n"
              << mpc_solution.u_star[500] << std::endl;
    std::cout << "Run time: " << mpc_solution.run_time_s << "s" << std::endl;
    std::cout << "Setup time: " << mpc_solution.setup_time_s << "s" << std::endl;
    std::cout << "Solve time: " << mpc_solution.solve_time_s << "s" << std::endl;

    return 0;
}