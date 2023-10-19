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

    OSQPInt setup_osqp(OSQP::Ptr osqp)
    {
        osqp->is_setup = true;
        return osqp_setup(&osqp->solver, osqp->P, osqp->q, osqp->A, osqp->l, osqp->u, osqp->m, osqp->n, osqp->settings);
    }

    OSQPInt solve_osqp(OSQP::Ptr osqp)
    {
        if (!osqp->is_setup)
            setup_osqp(osqp);
        return osqp_solve(osqp->solver);
    }

    OSQPInt update_settings(OSQP::Ptr osqp)
    {
        return osqp_update_settings(osqp->solver, osqp->settings);
    }

    void update_data(OSQP::Ptr osqp, QPProblem::Ptr qp)
    {
        if (qp->update.hessian)
        {
            to_csc(qp->hessian, osqp->P, osqp->Pnnz, osqp->Px, osqp->Pi, osqp->Pp);
            osqp_update_data_mat(osqp->solver, osqp->Px, osqp->Pi, osqp->Pnnz, OSQP_NULL, OSQP_NULL, OSQP_NULL);
            qp->update.hessian = false;
        }
        if (qp->update.gradient)
        {
            osqp->q = qp->gradient.data();
            osqp_update_data_vec(osqp->solver, osqp->q, OSQP_NULL, OSQP_NULL);
            qp->update.gradient = false;
        }
        if (qp->update.linear_constraint)
        {
            to_csc(qp->linear_constraint, osqp->A, osqp->Annz, osqp->Ax, osqp->Ai, osqp->Ap);
            osqp_update_data_mat(osqp->solver, osqp->Px, osqp->Pi, osqp->Pnnz, OSQP_NULL, OSQP_NULL, OSQP_NULL);
            qp->update.linear_constraint = false;
        }
        if (qp->update.lower_bound && qp->update.upper_bound)
        {
            osqp->u = qp->upper_bound.data();
            osqp->l = qp->lower_bound.data();
            osqp_update_data_vec(osqp->solver, OSQP_NULL, osqp->l, osqp->u);
            qp->update.lower_bound = false;
            qp->update.upper_bound = false;
        }
        if (qp->update.lower_bound)
        {
            osqp->l = qp->lower_bound.data();
            osqp_update_data_vec(osqp->solver, OSQP_NULL, osqp->l, OSQP_NULL);
            qp->update.lower_bound = false;
        }
        if (qp->update.upper_bound)
        {
            osqp->u = qp->upper_bound.data();
            osqp_update_data_vec(osqp->solver, OSQP_NULL, OSQP_NULL, osqp->u);
            qp->update.upper_bound = false;
        }
    }

    QPSolution::Ptr get_solution(OSQP::Ptr osqp)
    {
        QPSolution::Ptr qp_solution = std::make_shared<QPSolution>();
        qp_solution->xstar = Eigen::Map<EigenVector>(osqp->solver->solution->x, osqp->n);
        qp_solution->run_time_s = osqp->solver->info->run_time;
        qp_solution->setup_time_s = osqp->solver->info->setup_time;
        qp_solution->solve_time_s = osqp->solver->info->solve_time;
        return qp_solution;
    }

    template <auto ExecutionPolicy = std::execution::par>
    void solve_multi_osqp(std::vector<OSQP::Ptr> osqps)
    {
        std::for_each(ExecutionPolicy, osqps.begin(), osqps.end(), [&](OSQP::Ptr osqp) {
            solve_osqp(osqp);
        });
    }

    void run_mpc_osqp(OSQP::Ptr osqp, QPProblem::Ptr qp)
    {
        setup_osqp(osqp);
        while (osqp->ok)
        {
            solve_osqp(osqp);
            update_data(osqp, qp);
        }
    }

}