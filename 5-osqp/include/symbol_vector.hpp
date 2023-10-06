#ifndef ONR_OSQP_SYMBOL_VECTOR_HPP_
#define ONR_OSQP_SYMBOL_VECTOR_HPP_

#include <string>
#include <memory>
#include <ginac/ginac.h>

namespace boylan
{

    using namespace GiNaC;

    using symbol_ptr = std::shared_ptr<symbol>;
    
    class symbol_vector : public std::vector<symbol_ptr>
    {
        public:
            symbol_vector()
            {}

            symbol_vector(const std::string id, const int n)
            : std::vector<symbol_ptr>(n)
            {
                for (int i = 0; i < n; i++)
                {
                    this->at(i) = std::make_shared<symbol>(id + std::to_string(i));
                }
            }

            const symbol& operator[](const int i) const
            {
                return *(this->at(i));
            }

            symbol_vector append(const symbol_vector& vec2)
            {
                symbol_vector vec;
                vec.insert(vec.end(), this->begin(), this->end());
                vec.insert(vec.end(), vec2.begin(), vec2.end());
                return vec;
            }
    };

}

#endif