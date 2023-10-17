#include "orlqp/osqp_util.hpp"

namespace orlqp
{

    OSQP::Ptr qp2osqp(const QPProblem::Ptr qp)
    {
        OSQP::Ptr osqp = std::make_shared<OSQP>();
        osqp->settings = new OSQPSettings;
        osqp_set_default_settings(osqp->settings);

        osqp->n = qp->num_variables;
        osqp->m = qp->num_constraints;
        osqp->q = qp->gradient.data();
        osqp->l = qp->lower_bound.data();
        osqp->u = qp->upper_bound.data();

        to_csc(qp->hessian, osqp->P, osqp->Pnnz, osqp->Px, osqp->Pi, osqp->Pp);

        to_csc(qp->linear_constraint, osqp->A, osqp->Annz, osqp->Ax, osqp->Ai, osqp->Ap);

        osqp_setup(&osqp->solver, osqp->P, osqp->q, osqp->A, osqp->l, osqp->u, osqp->m, osqp->n, osqp->settings);

        return osqp;
    }

    void to_csc(const EigenSparseMatrix &matrix,
                OSQPCscMatrix *&M, OSQPInt &Mnnz, OSQPFloat *&Mx, OSQPInt *&Mi, OSQPInt *&Mp)
    {
        M = new OSQPCscMatrix;
        Mnnz = matrix.nonZeros();
        Mx = new OSQPFloat[Mnnz];
        Mi = new OSQPInt[Mnnz];
        Mp = new OSQPInt[matrix.cols() + 1];

        int k = 0;
        Mp[0] = 0;
        for (int j = 0; j < matrix.outerSize(); ++j)
        {
            for (EigenSparseMatrix::InnerIterator it(matrix, j); it; ++it)
            {
                Mx[k] = it.value();
                Mi[k] = it.row();
                ++k;
            }
            Mp[j + 1] = k;
        }
        csc_set_data(M, matrix.rows(), matrix.cols(), Mnnz, Mx, Mi, Mp);
    }

    OSQPInt solve_osqp(OSQP::Ptr osqp)
    {
        return osqp_solve(osqp->solver);
    }

    OSQPInt update_settings(OSQP::Ptr osqp)
    {
        return osqp_update_settings(osqp->solver, osqp->settings);
    }

}