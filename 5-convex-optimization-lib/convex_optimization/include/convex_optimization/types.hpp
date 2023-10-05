#ifndef CONVEX_OPTIMIZATION_TYPES_HPP_
#define CONVEX_OPTIMIZATION_TYPES_HPP_

#include <ginac/ginac.h>
#include <memory>

namespace boylan
{

    using namespace GiNaC;

    using symbol_ptr = std::shared_ptr<symbol>;
    using ex_ptr = std::shared_ptr<ex>;
    using matrix_ptr = std::shared_ptr<matrix>;

    class symbol_vector : public std::vector<symbol_ptr>
    {
    public:
        symbol_vector(const std::string id, const int n)
            : std::vector<symbol_ptr>(n)
        {
            for (int i = 0; i < n; i++)
            {
                this->at(i) = std::make_shared<symbol>(id + std::to_string(i));
            }
        }
    };

}

#endif