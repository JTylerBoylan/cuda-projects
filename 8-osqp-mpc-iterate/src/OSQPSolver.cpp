#include "OSQPSolver.hpp"

namespace boylan
{

    OSQP::OSQP()
    {
        solver_ = std::make_shared<OSQPSolver *>();
        settings_ = std::make_shared<OSQPSettings>();
        solution_ = std::make_shared<OSQPSolution>();
    }

    bool OSQP::setup(QPProblem &problem)
    {
        const int n = problem.variableCount();
        const int m = problem.constraintCount();
        hessian_csc_ = convertSparseMatrixToCSC(problem.getHessianMatrix());
        OSQPFloat *q = problem.getGradientVector().data();
        problem.getLinearConstraintMatrix().makeCompressed();
        lin_constraint_csc_ = convertSparseMatrixToCSC(problem.getLinearConstraintMatrix());
        OSQPFloat *l = problem.getLowerBoundVector().data();
        OSQPFloat *u = problem.getUpperBoundVector().data();
        return osqp_setup(solver_.get(), hessian_csc_.get(), q, lin_constraint_csc_.get(), l, u, m, n, settings_.get());
    }

    bool OSQP::solve(QPProblem &problem)
    {
        latest_exit_ = osqp_solve(*solver_.get());
        latest_solution_.x_star = Eigen::Map<EigenVector>(solution_->x, problem.variableCount());
        return latest_exit_ == 0;
    }

    OSQP::~OSQP()
    {
        osqp_cleanup(*solver_.get());
        if (hessian_csc_)
            freeCSC(hessian_csc_);
        if (lin_constraint_csc_)
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
        delete matrix->x;
        delete matrix->i;
        delete matrix->p;
    }

}
