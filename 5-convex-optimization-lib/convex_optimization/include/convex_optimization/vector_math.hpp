#ifndef CONVEX_OPTIMIZATION_VECTOR_MATH_HPP_
#define CONVEX_OPTIMIZATION_VECTOR_MATH_HPP_

#include <convex_optimization/types.hpp>
#include <assert.h>

namespace boylan
{

    matrix jacobian(ex f, symbol_vector x)
    {
        matrix jacob(x.size(), 1);
        for (int i = 0; i < x.size(); i++)
        {
            jacob(i,0) = f.diff(*x[i]);
        }
        return jacob;
    }

    matrix jacobian(matrix f, symbol_vector x)
    {
        assert(("f must be a scalar or vector.", f.cols() == 1));
        matrix jacob(f.rows(), x.size());
        for (int r = 0; r < f.rows(); r++)
        {
            for (int c = 0; c < x.size(); c++)
            {
                jacob(r,c) = f(r,0).diff(*x[c]);
            }
        }
        return jacob;
    }

    matrix hessian(ex f, symbol_vector x)
    {
        return jacobian(jacobian(f,x),x);
    }

    ex dot(matrix a, matrix b)
    {
        assert(("a & b must be column vectors", a.cols() == 1 && b.cols() == 1));
        assert(("a & b must be the same size", a.rows() == b.rows()));
        ex sum;
        for (int i = 0; i < a.rows(); i++)
        {
            sum += a(i,0)*b(i,0);
        }
        return sum;
    }

}

#endif