#include <iostream>

#include "MPCObjective.hpp"
#include "MPCDynamics.hpp"
#include "MPCConstraints.hpp"
#include "MPCSolver.hpp"

#include "osqp/osqp.h"

#define NUMBER_OF_NODES 10
#define HORIZON_TIME 1.0F

using namespace boylan;

int main()
{

    int N = NUMBER_OF_NODES;
    float deltaT = HORIZON_TIME / (float)NUMBER_OF_NODES;

    int Nx = 2;
    int Nu = 1;

    EigenMatrix Q(Nx, Nx), R(Nu, Nu);
    Q << 1.0, 0.0, 1.0, 0.0;
    R << 1.0;
    EigenVector xf(Nx);
    xf << 0.0, 0.0;
    MPCObjective::Ptr objective = std::make_shared<MPCObjective>(N, Nx, Nu, Q, R, xf);

    EigenMatrix A(Nx, Nx), B(Nx, Nu);
    A << 1, deltaT, 0, 1;
    B << 0, deltaT;
    MPCDynamics::Ptr dynamics = std::make_shared<MPCDynamics>(N, Nx, Nu, A, B);

    EigenVector x0(Nx);
    x0 << 1.0, 1.0;
    EigenVector x_min(Nx), x_max(Nx), u_min(Nu), u_max(Nu);
    x_min << -10.0, -10.0;
    x_max << +10.0, +10.0;
    u_min << -2.0;
    u_max << +2.0;
    MPCConstraints::Ptr constraints = std::make_shared<MPCConstraints>(N, Nx, Nu, x0, x_min, x_max, u_min, u_max);

    int n = Nx * (N + 1) + Nu * N;
    int m = 2 * Nx * (N + 1) + Nu * N;

    MPCSolver::Ptr solver = std::make_shared<MPCSolver>(n, m, objective, dynamics, constraints);
    solver->getSettings()->warm_starting = true;
    solver->getSettings()->verbose = true;

    OSQPInt exitflag = solver->solve();

    OSQPSolution *solution = solver->getSolution();

    if (exitflag == 0)
    {
        for (OSQPInt i = 0; i < N; ++i)
        {
            const int idx = 2 * i;
            std::cout << "x[" << i << "] = " << solution->x[idx] << std::endl;
            std::cout << "v[" << i << "] = " << solution->x[idx + 1] << std::endl;
        }
        for (OSQPInt i = 0; i < N - 1; ++i)
        {
            std::cout << "u[" << i << "] = " << solution->x[2 * N + i] << std::endl;
        }
    }
    else
    {
        std::cerr << "OSQP solve failed with exit flag " << exitflag << std::endl;
    }

    return exitflag;
}