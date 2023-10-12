#include "OSQPSolver.hpp"

namespace boylan
{

    OSQP::OSQP()
    {
        osqp_solver_ = std::make_shared<OSQPSolver *>();
        osqp_settings_ = std::make_shared<OSQPSettings>();
        osqp_solution_ = std::make_shared<OSQPSolution *>();

        osqp_set_default_settings(osqp_settings_.get());
    }

    bool OSQP::setup(QPProblem &problem)
    {
        const int n = problem.variableCount();
        const int m = problem.constraintCount();

        freeCSC(hessian_csc_);
        hessian_csc_ = convertSparseMatrixToCSC(problem.getHessianMatrix());

        OSQPFloat *q = problem.getGradientVector().data();

        problem.getLinearConstraintMatrix().makeCompressed();
        freeCSC(lin_constraint_csc_);
        lin_constraint_csc_ = convertSparseMatrixToCSC(problem.getLinearConstraintMatrix());

        OSQPFloat *l = problem.getLowerBoundVector().data();
        OSQPFloat *u = problem.getUpperBoundVector().data();

        return osqp_setup(osqp_solver_.get(), hessian_csc_.get(), q, lin_constraint_csc_.get(), l, u, m, n, osqp_settings_.get());
    }

    bool OSQP::solve(QPProblem &problem)
    {
        latest_exit_ = osqp_solve(*osqp_solver_.get());
        *osqp_solution_ = (*osqp_solver_.get())->solution;
        if (latest_exit_ == 0) {
            qp_solution_.x_star = Eigen::Map<EigenVector>((*osqp_solution_)->x, problem.variableCount());
            qp_solution_.run_time_s = (*osqp_solver_.get())->info->run_time;
            qp_solution_.setup_time_s = (*osqp_solver_.get())->info->setup_time;
            qp_solution_.solve_time_s = (*osqp_solver_.get())->info->solve_time;
        }
        return latest_exit_ == 0;
    }

    OSQP::~OSQP()
    {
        osqp_cleanup(*osqp_solver_.get());
        freeCSC(hessian_csc_);
        freeCSC(lin_constraint_csc_);
    }

    std::shared_ptr<OSQPCscMatrix> OSQP::convertSparseMatrixToCSC(const EigenSparseMatrix &matrix)
    {
        OSQPInt A_nnz = matrix.nonZeros();
        OSQPFloat *A_x = new OSQPFloat[A_nnz];
        OSQPInt *A_i = new OSQPInt[A_nnz];
        OSQPInt *A_p = new OSQPInt[matrix.cols() + 1];

        int k = 0;
        A_p[0] = 0;
        for (int j = 0; j < matrix.outerSize(); ++j)
        {
            for (EigenSparseMatrix::InnerIterator it(matrix, j); it; ++it)
            {
                A_x[k] = it.value();
                A_i[k] = it.row();
                ++k;
            }
            A_p[j + 1] = k;
        }
        std::shared_ptr<OSQPCscMatrix> csc = std::make_shared<OSQPCscMatrix>();
        csc_set_data(csc.get(), matrix.rows(), matrix.cols(), A_nnz, A_x, A_i, A_p);
        return csc;
    }

    void OSQP::freeCSC(std::shared_ptr<OSQPCscMatrix> matrix)
    {
        if (matrix)
        {
            delete matrix->x;
            delete matrix->i;
            delete matrix->p;
        }
    }

}
