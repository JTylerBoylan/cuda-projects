#ifndef CONVEX_OPTIMIZATION_INDEX_VECTOR_HPP_
#define CONVEX_OPTIMIZATION_INDEX_VECTOR_HPP_

#include <stdlib.h>
#include <string>
#include <memory>
#include <ginac/ginac.h>

namespace boylan
{

using namespace GiNaC;

using symbolPtr_t = std::shared_ptr<symbol>;

class IndexVector
{
public:

    IndexVector(const std::string& id, int n)
    {
        syms.resize(n);
        for (int idx = 0; idx < n; idx++)
        {
            syms[idx] = std::make_shared<symbol>(id + std::to_string(idx));
        }
    }

    inline symbol& i(int idx)
    {
        return *(syms[idx]);
    }

    inline size_t size() const
    {
        return syms.size();
    }

private:

    std::vector<symbolPtr_t> syms;

};

    
}

#endif