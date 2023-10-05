#ifndef CONVEX_OPTIMIZATION_INTERIOR_POINT_SOLVER_HPP_
#define CONVEX_OPTIMIZATION_INTERIOR_POINT_SOLVER_HPP_

#include <convex_optimization/types.hpp>
#include <convex_optimization/vector_math.hpp>
#include <assert.h>

namespace boylan
{

    class InteriorPointMethod
    {

    public:
        InteriorPointMethod(const symbol_vector x, const ex_ptr objective, const matrix_ptr constraints,
                            const matrix_ptr A, const matrix_ptr b)
            : x_(x), objective_(objective), constraints_(constraints), A_(A), b_(b),
              n_(x.size()), m_(constraints->rows()), p_(b->rows()),
              lam_(symbol_vector("lam", m_)), nu_(symbol_vector("nu", p_))
        {
            assert(("constraints must be size [mx1]", constraints->cols() == 1));
            assert(("b must be size [px1]", b->cols() == 1));
            assert(("A must be size [pxn]", A->rows() == p_ && A->cols() == n_));
        }

        const symbol_vector &x()
        {
            return x_;
        }

        ex &objective()
        {
            return *objective_;
        }

        matrix &constraints()
        {
            return *constraints_;
        }

        matrix &A()
        {
            return *A_;
        }

        matrix &b()
        {
            return *b_;
        }

    private:
        const symbol_vector x_;

        ex_ptr objective_;
        matrix_ptr constraints_;
        matrix_ptr A_;
        matrix_ptr b_;

        size_t n_, m_, p_;

        symbol_vector lam_, nu_;
    };

}

#endif