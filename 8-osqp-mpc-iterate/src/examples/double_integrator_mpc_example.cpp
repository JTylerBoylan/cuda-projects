#include "DoubleIntegratorMPCProblem.hpp"
#include "OSQPSolver.hpp"

#include <iostream>

using namespace boylan;

int main()
{

    OSQP solver;
    solver.getSettings()->polishing = true;
    solver.getSettings()->warm_starting = true;
    solver.getSettings()->verbose = true;

    DoubleIntegratorMPCProblem problem;
    problem.setup();

    solver.setup(problem);

    solver.solve(problem);
    auto solution = problem.QPtoDoubleIntegratorSolution(solver.getSolution());

    std::cout << "X star:\n" << solution.x_star << std::endl;
    std::cout << "V star:\n" << solution.v_star << std::endl;
    std::cout << "U star:\n" << solution.u_star << std::endl;
    std::cout << "Run time: " << solution.run_time_s << "s" << std::endl;
    std::cout << "Setup time: " << solution.setup_time_s << "s" << std::endl;
    std::cout << "Solve time: " << solution.solve_time_s << "s" << std::endl;

    return 0;
}