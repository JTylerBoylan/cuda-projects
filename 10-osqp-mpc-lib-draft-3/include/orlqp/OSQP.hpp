#ifndef ORLQP_OSQP_SOLVER_HPP_
#define ORLQP_OSQP_SOLVER_HPP_

#include <execution>

#include "orlqp/types.hpp"
#include "osqp/osqp.h"
#include "orlqp/QPProblem.hpp"

namespace orlqp
{

    class OSQP
    {
    public:
        using Ptr = std::shared_ptr<OSQP>;

        OSQP();

        ~OSQP()
        {
            osqp_cleanup(solver);
            delete P;
            delete A;
            delete settings;
        }

        OSQPInt solve();

        OSQPInt setupFromQP(const QPProblem::Ptr qp);

        OSQPInt updateFromQP(const QPProblem::Ptr qp);

        OSQPInt updateSettings();

        QPSolution::Ptr getQPSolution();

    private:
        OSQPInt n, m;

        OSQPSolver *solver = nullptr;
        OSQPSettings *settings = nullptr;

        bool is_setup = false;
        bool ok = true;

        OSQPFloat *q = nullptr;
        OSQPFloat *l = nullptr;
        OSQPFloat *u = nullptr;

        OSQPCscMatrix *P = nullptr;
        OSQPInt Pnnz;
        OSQPFloat *Px = nullptr;
        OSQPInt *Pi = nullptr;
        OSQPInt *Pp = nullptr;

        OSQPCscMatrix *A = nullptr;
        OSQPInt Annz;
        OSQPFloat *Ax = nullptr;
        OSQPInt *Ai = nullptr;
        OSQPInt *Ap = nullptr;

        void convertEigenSparseToCSC(const EigenSparseMatrix &matrix,
                                     OSQPCscMatrix *&M, OSQPInt &Mnnz, OSQPFloat *&Mx, OSQPInt *&Mi, OSQPInt *&Mp);
    };

}

#endif