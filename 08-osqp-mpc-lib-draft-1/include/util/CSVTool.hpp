#ifndef ORLQP_CSV_TOOL_HPP_
#define ORLQP_CSV_TOOL_HPP_

#include <string>
#include <fstream>

#include "QPTypes.hpp"
#include "QPProblem.hpp"

namespace boylan
{
    void clearFile(const std::string file_name)
    {
        std::ofstream file;
        file.open(file_name);
        file.clear();
        file.close();
    }

    void appendQPSolutionToCSV(const int i, const QPSolution &solution, const std::string file_name)
    {
        std::ofstream file;
        file.open(file_name, std::ofstream::out | std::fstream::app);

        file << i << ",";
        file << solution.run_time_s << ","
             << solution.setup_time_s << ","
             << solution.solve_time_s << ",";

        file << solution.x_star.rows();
        for (int i = 0; i < solution.x_star.rows(); i++)
        {
            file << "," << solution.x_star(i);
        }
        file << "\n";
        file.close();
    }

}

#endif