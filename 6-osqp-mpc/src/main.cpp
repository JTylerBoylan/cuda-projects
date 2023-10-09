#include "MPCObjective.hpp"
#include "MPCDynamics.hpp"
#include "MPCConstraints.hpp"

#include <OsqpEigen/OsqpEigen.h>

#define NUMBER_OF_NODES 5
#define HORIZON_TIME 1.0F

using namespace boylan;

int main()
{

    int N = NUMBER_OF_NODES;
    float deltaT = (float)NUMBER_OF_NODES / HORIZON_TIME;

    int Nx = 2;
    int Nu = 1;

    MATRIX A(Nx, Nx), B(Nx, Nu);
    A << 1, deltaT, 0, 1;
    B << 0, deltaT;
    MPCDynamics dynamics(N, Nx, Nu, A, B);

    VECTOR x0(Nx);
    x0 << 1.0, 1.0;
    VECTOR x_min(Nx), x_max(Nx), u_min(Nu), u_max(Nu);
    x_min << -10.0, -10.0;
    x_max << +10.0, +10.0;
    u_min << -2.0;
    u_max << +2.0;
    MPCConstraints constraints(N, Nx, Nu, x0, x_min, x_max, u_min, u_max);

    MATRIX Q(Nx, Nx), R(Nu, Nu);
    Q << 1.0, 0.0, 1.0, 0.0;
    R << 1.0;
    VECTOR xf(Nx);
    xf << 0.0, 0.0;
    MPCObjective objective(N, Nx, Nu, Q, R, xf);

    int n = Nx * (N + 1) + Nu * N;
    int m = 2 * Nx * (N + 1) + Nu * N;

    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(true);
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables(n);
    solver.data()->setNumberOfConstraints(m);
    solver.data()->setGradient(objective.getGradient());
    solver.data()->setHessianMatrix(objective.getHessian());
    solver.data()->setLinearConstraintsMatrix(dynamics.getLinearConstraintMatrix());
    solver.data()->setBounds(constraints.getLowerBounds(), constraints.getUpperBounds());

    solver.initSolver();

    auto exit = solver.solveProblem();


    return 0;
}