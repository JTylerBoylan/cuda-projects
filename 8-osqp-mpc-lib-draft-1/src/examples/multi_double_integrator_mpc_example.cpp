#include "double_integrator_mpc/DoubleIntegratorMPCModel.hpp"
#include "osqp/OSQPSolver.hpp"
#include "mpc/MultiMPCProblem.hpp"

#include "util/CSVTool.hpp"

#include <iostream>
#include <fstream>

using namespace boylan;

int main()
{

    const size_t maxN = 5000;
    const size_t dN = 50;

    const std::string result_file = "/app/results/multi_mpc_benchmark.csv";
    clearFile(result_file);

    for (size_t N = dN; N <= maxN; N += dN)
    {

        EigenVector initial_state(2), desired_state(2);
        initial_state << 2.0, 0.0;
        desired_state << -1.0, 0.0;

        MultiMPCProblem<DoubleIntegratorMPCModel> problem(N);

        for (auto iter = problem.beginIndex(); iter != problem.endIndex(); ++iter)
        {
            problem.getSolver(*iter)->getSettings()->verbose = false;
        }

        problem.setup();

        for (auto iter = problem.beginIndex(); iter != problem.endIndex(); ++iter)
        {
            problem.updateInitialState(*iter, initial_state);
            problem.updateDesiredState(*iter, desired_state);
        }

        problem.solve();

        QPSolution sum;
        for (auto iter = problem.beginIndex(); iter != problem.endIndex(); ++iter)
        {
            MPCSolution &mpc_solution = problem.getSolution(*iter);
            DoubleIntegratorMPCSolution solution = problem.getModel(*iter)->MPCtoDoubleIntegratorSolution(mpc_solution);
            sum.run_time_s += solution.run_time_s;
            sum.setup_time_s += solution.setup_time_s;
            sum.solve_time_s += solution.solve_time_s;
            if (*iter == 0)
            {
                sum.x_star = solution.x_star;
            }
        }

        appendQPSolutionToCSV(N, sum, result_file);
        std::cout << N << "\n";
    }

    return 0;
}