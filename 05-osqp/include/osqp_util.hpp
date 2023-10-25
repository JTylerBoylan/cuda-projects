#ifndef ONR_OSQP_UTIL_HPP_
#define ONR_OSQP_UTIL_HPP_

#include <ginac/ginac.h>
#include <osqp/osqp.h>

namespace boylan
{

    using namespace GiNaC;

    void toCSC(matrix M, OSQPCscMatrix *cscM)
    {
        int numRows = M.rows();
        int numCols = M.cols();

        // Count non-zeros in each column
        std::vector<int> col_ptr(numCols + 1, 0);
        int nnz = 0; // Count of non-zero entries
        for (int j = 0; j < numCols; ++j)
        {
            for (int i = 0; i < numRows; ++i)
            {
                if (M(i, j) != 0)
                {
                    col_ptr[j + 1]++;
                    nnz++;
                }
            }
        }

        // Compute column pointers
        for (int j = 1; j <= numCols; ++j)
        {
            col_ptr[j] += col_ptr[j - 1];
        }

        // Fill row indices and values
        std::vector<int> row_idx(nnz);
        std::vector<float> values(nnz);
        std::vector<int> counters = col_ptr; // Make a copy to keep track of where to insert data
        int counter = 0;
        for (int j = 0; j < numCols; ++j)
        {
            for (int i = 0; i < numRows; ++i)
            {
                if (M(i, j) != 0)
                {
                    row_idx[counter] = i;
                    values[counter] = GiNaC::ex_to<GiNaC::numeric>(M(i, j)).to_double(); // Assuming the matrix contains numerics
                    counter++;
                }
            }
        }

        // Construct the OSQPCscMatrix
        cscM->m = numRows;
        cscM->n = numCols;
        cscM->nz = nnz;
        cscM->nzmax = nnz;
        cscM->p = col_ptr.data();
        cscM->i = row_idx.data();
        cscM->x = values.data();
    }

    std::vector<float> toVector(const matrix M)
    {
        std::vector<float> vec(M.rows());
        for (int i = 0; i < M.rows(); i++)
        {
            vec[i] = ex_to<numeric>(M(i, 0)).to_double();
        }
        return vec;
    }

}

#endif