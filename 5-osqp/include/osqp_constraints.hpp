#ifndef ONR_OSQP_CONSTRAINTS_HPP_
#define ONR_OSQP_CONSTRAINTS_HPP_

#include "symbol_vector.hpp"
#include <tuple>

namespace boylan
{

    class osqp_constraints : public std::vector<std::tuple<ex, ex, ex>>
    {
    public:
        osqp_constraints(const symbol_vector x, const int n)
            : std::vector<std::tuple<ex, ex, ex>>(n), symbols(x)
        {
        }

        void add_constraints(ex constraint, ex lower_bound, ex upper_bound)
        {
            this->push_back(std::tuple<ex, ex, ex>{constraint, lower_bound, upper_bound});
        }

        matrix get_A()
        {
            matrix A(this->size(), symbols.size());
            for (int r = 0; r < this->size(); r++)
            {
                for (int c = 0; c < symbols.size(); c++)
                {
                    A(r, c) = std::get<0>(this->at(r)).diff(symbols[c]);
                }
            }
            return A;
        }

        matrix get_l()
        {
            matrix l(this->size(), 1);
            for (int i = 0; i < this->size(); i++)
            {
                l(i,0) = std::get<1>(this->at(i));
            }
            return l;
        }

        matrix get_u()
        {
            matrix u(this->size(), 1);
            for (int i = 0; i < this->size(); i++)
            {
                u(i,0) = std::get<2>(this->at(i));
            }
            return u;
        }

    private:
        const symbol_vector symbols;
    };

}

#endif