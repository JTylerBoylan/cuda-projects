#ifndef ORLQP_OSQP_SOLVER_HPP_
#define ORLQP_OSQP_SOLVER_HPP_

#include "orlqp/types.hpp"
#include "osqp/osqp.h"

namespace orlqp
{

    struct OSQP
    {

        using Ptr = std::shared_ptr<OSQP>;

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

        ~OSQP()
        {
            osqp_cleanup(solver);
            delete P;
            delete A;
            delete settings;
        }
    };

}

#endif