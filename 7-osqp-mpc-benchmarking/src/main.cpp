#include <iostream>

#include "MPCObjective.hpp"
#include "MPCDynamics.hpp"
#include "MPCConstraints.hpp"
#include "MPCSolver.hpp"

#include "osqp/osqp.h"

#define NUMBER_OF_NODES 11
#define HORIZON_TIME 1.0F

#define INTERATION_STEPS 50

using namespace boylan;

int main()
{

    int N = NUMBER_OF_NODES;
    float deltaT = HORIZON_TIME / (float)NUMBER_OF_NODES;

    int Nx = 2;
    int Nu = 1;

    EigenMatrix Q(Nx, Nx), R(Nu, Nu);
    Q << 1.0, 0.0,
        0.0, 0.0;
    R << 0.0;
    EigenVector xf(Nx);
    xf << 0.0, 0.0;
    MPCObjective::Ptr objective = std::make_shared<MPCObjective>(N, Nx, Nu, Q, R, xf);

    EigenMatrix A(Nx, Nx), B(Nx, Nu);
    A << 1, deltaT, 0, 1;
    B << 0, deltaT;
    MPCDynamics::Ptr dynamics = std::make_shared<MPCDynamics>(N, Nx, Nu, A, B);

    EigenVector x0(Nx);
    x0 << 1.0, 0.0;
    EigenVector x_min(Nx), x_max(Nx), u_min(Nu), u_max(Nu);
    x_min << -10.0, -10.0;
    x_max << +10.0, +10.0;
    u_min << -5.0;
    u_max << +5.0;
    MPCConstraints::Ptr constraints = std::make_shared<MPCConstraints>(N, Nx, Nu, x0, x_min, x_max, u_min, u_max);

    int n = Nx * (N + 1) + Nu * N;
    int m = 2 * Nx * (N + 1) + Nu * N;

    MPCSolver::Ptr solver = std::make_shared<MPCSolver>(n, m, objective, dynamics, constraints);
    solver->getSettings()->max_iter = 10000;
    solver->getSettings()->warm_starting = true;
    solver->getSettings()->verbose = false;
    solver->getSettings()->polishing = false;

    solver->setup();

    EigenVector z_star, u_star;
    for (int k = 0; k < INTERATION_STEPS; k++)
    {
        std::cout << "x0(" << k << ") = \n" << x0 << "\n";
        if (solver->solve() != 0)
        {
            std::cout << "Solver failed on iteration " << k << "\n";
            break;
        }

        z_star = Eigen::Map<EigenVector>(solver->getSolution()->x, n);
        u_star = z_star.block(Nx * (N + 1), 0, Nu, 1);

        x0 = A * x0 + B * u_star;

        solver->updateX0(x0);

        OSQPFloat time_s = solver->getSolver()->info->run_time;

        std::cout << "Computation Time:" << time_s << "\n";
    }

    return 0;
}