#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <execution>
#include <mutex>

#include "MPCObjective.hpp"
#include "MPCDynamics.hpp"
#include "MPCConstraints.hpp"
#include "MPCSolver.hpp"

#include "osqp/osqp.h"

#define MAX_NUMBER_OF_PROBLEMS 10000
#define NUMBER_OF_PROBLEMS_INCREMENT 100

#define NUM_THREADS 10

#define NUM_STATES 2
#define NUM_CONTROLS 1

#define NUMBER_OF_NODES 11
#define HORIZON_TIME 1.0F

#define INTERATION_STEPS 50

using namespace boylan;

EigenMatrix getQ(const int n_problem, const int Nx)
{
    EigenMatrix Q(Nx, Nx);
    Q << 1.0, 0.0,
        0.0, 0.0;
    return Q;
}

EigenMatrix getR(const int n_problem, const int Nu)
{
    EigenMatrix R(Nu, Nu);
    R << 0.0;
    return R;
}

EigenVector getXf(const int n_problem, const int Nx)
{
    EigenVector xf(Nx);
    xf << 0.0, 0.0;
    return xf;
}

EigenMatrix getA(const int n_problem, const int Nx, const float deltaT)
{
    EigenMatrix A(Nx, Nx);
    A << 1, deltaT, 0, 1;
    return A;
}

EigenMatrix getB(const int n_problem, const int Nx, const int Nu, const float deltaT)
{
    EigenMatrix B(Nx, Nu);
    B << 0, deltaT;
    return B;
}

EigenVector getX0(const int n_problem, const int Nx)
{
    EigenVector x0(Nx);
    x0 << 1.0, 0.0;
    return x0;
}

EigenVector getXmin(const int n_problem, const int Nx)
{
    EigenVector x_min(Nx);
    x_min << -10.0, -10.0;
    return x_min;
}

EigenVector getXmax(const int n_problem, const int Nx)
{
    EigenVector x_max(Nx);
    x_max << +10.0, +10.0;
    return x_max;
}

EigenVector getUmin(const int n_problem, const int Nu)
{
    EigenVector u_min(Nu);
    u_min << -5.0;
    return u_min;
}

EigenVector getUmax(const int n_problem, const int Nu)
{
    EigenVector u_max(Nu);
    u_max << +5.0;
    return u_max;
}

int main()
{

    std::ofstream results_file;
    results_file.open("/app/results_CPU3.csv");

    const int Nx = NUM_STATES;
    const int Nu = NUM_CONTROLS;
    const int N = NUMBER_OF_NODES;
    const float deltaT = HORIZON_TIME / (float)NUMBER_OF_NODES;

    for (int NUMBER_OF_PROBLEMS = NUMBER_OF_PROBLEMS_INCREMENT;
         NUMBER_OF_PROBLEMS < MAX_NUMBER_OF_PROBLEMS;
         NUMBER_OF_PROBLEMS += NUMBER_OF_PROBLEMS_INCREMENT)
    {

        std::cout << NUMBER_OF_PROBLEMS << "\n";
        results_file << NUMBER_OF_PROBLEMS << ",";

        const int NUMBER_OF_PROBLEMS_PER_THREAD = NUMBER_OF_PROBLEMS / NUM_THREADS;

        std::vector<int> idx(NUM_THREADS);
        std::iota(idx.begin(), idx.end(), 0);

        float solve_time = 0.0f;
        std::mutex mtx;
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](const int index)
        {
            const int startIdx = NUMBER_OF_PROBLEMS_PER_THREAD*index;
            MPCObjective::Ptr objective = std::make_shared<MPCObjective>(NUMBER_OF_PROBLEMS_PER_THREAD, N, Nx, Nu);
            MPCDynamics::Ptr dynamics = std::make_shared<MPCDynamics>(NUMBER_OF_PROBLEMS_PER_THREAD, N, Nx, Nu);
            MPCConstraints::Ptr constraints = std::make_shared<MPCConstraints>(NUMBER_OF_PROBLEMS_PER_THREAD, N, Nx, Nu);

            for (int n_prob = 0; n_prob < NUMBER_OF_PROBLEMS_PER_THREAD; ++n_prob)
            {
                const EigenMatrix Q = getQ(startIdx + n_prob, Nx);
                const EigenMatrix R = getR(startIdx + n_prob, Nu);
                const EigenVector xf = getXf(startIdx + n_prob, Nx);
                const EigenMatrix A = getA(startIdx + n_prob, Nx, deltaT);
                const EigenMatrix B = getB(startIdx + n_prob, Nx, Nu, deltaT);
                const EigenVector x0 = getX0(startIdx + n_prob, Nx);
                const EigenVector x_min = getXmin(startIdx + n_prob, Nx);
                const EigenVector x_max = getXmax(startIdx + n_prob, Nx);
                const EigenVector u_min = getUmin(startIdx + n_prob, Nu);
                const EigenVector u_max = getUmax(startIdx + n_prob, Nu);

                objective->setObjective(n_prob, Q, R, xf);
                dynamics->setDynamics(n_prob, A, B);
                constraints->setConstraints(n_prob, x0, x_min, x_max, u_min, u_max);
            }

            int n = NUMBER_OF_PROBLEMS_PER_THREAD * (Nx * (N + 1) + Nu * N);
            int m = NUMBER_OF_PROBLEMS_PER_THREAD * (2 * Nx * (N + 1) + Nu * N);

            MPCSolver::Ptr solver = std::make_shared<MPCSolver>(n, m, objective, dynamics, constraints);
            solver->getSettings()->max_iter = 10000;
            solver->getSettings()->warm_starting = true;
            solver->getSettings()->verbose = false;
            solver->getSettings()->polishing = false;

            solver->setup();

            EigenVector z_star, u_star, x0;
            if (solver->solve() != 0)
            {
                std::cout << "Solver failed.\n";
                return 1;
            }

            z_star = Eigen::Map<EigenVector>(solver->getSolution()->x, n);
            u_star = z_star.block(Nx * (N + 1), 0, Nu, 1);

            // std::cout << z_star << "\n";

            OSQPFloat time_run_s = solver->getSolver()->info->run_time;

            std::lock_guard<std::mutex> guard(mtx);
            solve_time += time_run_s;

            return 0;
        });

        std::cout << "Run Time: " << solve_time << " sec\n";
        results_file << solve_time << "\n";
    }

    results_file.close();

    return 0;
}