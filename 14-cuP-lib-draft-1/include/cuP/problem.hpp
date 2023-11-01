#ifndef CUP_PROBLEM_HPP_
#define CUP_PROBLEM_HPP_

#include "cuP/types.hpp"

namespace cuP
{

    class cuPProblem
    {
    public:
        cuPProblem()
        {
        }

        ex objective_function(std::vector<symbol_ptr> &x, std::vector<symbol_ptr> &weights, const int index)
        {
            ex objective;
            for (int i = 0; i < x.size(); i++)
            {
                objective += (*weights[i]) * (*x[i]) * (*x[i]);
            }
            return objective;
        }

        matrix inequality_constraints(std::vector<symbol_ptr> &x, std::vector<symbol_ptr> &coeffs, const int index)
        {
            /**
             * TODO:
            */
        }

        std::vector<float> initial_variables(const int index)
        {
            /**
             * TODO:
            */
        }
    };

}

#endif