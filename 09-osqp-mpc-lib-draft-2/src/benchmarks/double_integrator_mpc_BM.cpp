#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <fstream>

#include "orlqp/mpc_util.hpp"
#include "orlqp/osqp_util.hpp"

#define NUMBER_OF_STATES 2
#define NUMBER_OF_CONTROLS 1

#define POSITION_ERROR_COST_WEIGHT 10.0
#define VELOCITY_ERROR_COST_WEIGHT 1.0
#define FORCE_COST_WEIGHT 0.0

#define HORIZON_TIME 1.0
#define MASS 1.0

#define MIN_POSITION -10.0
#define MAX_POSITION +10.0
#define MIN_VELOCITY -10.0
#define MAX_VELOCITY +10.0
#define MIN_FORCE -5.0
#define MAX_FORCE +5.0

#define NUMBER_OF_MPC_ITERATIONS 100
#define MIN_NUMBER_OF_NODES 1
#define MAX_NUMBER_OF_NODES 100

using namespace orlqp;

int main()
{
    srand(time(NULL));

    std::ofstream out_file;
    out_file.open("/app/results.csv");

    for (int Nn = MIN_NUMBER_OF_NODES; Nn <= MAX_NUMBER_OF_NODES; Nn++)
    {

        out_file << Nn << ",";

        EigenVector x0(NUMBER_OF_STATES), xf(NUMBER_OF_STATES);
        x0 << 5.0, 0.0;
        xf << 0.0, 0.0;

        auto mpc = create_mpc(NUMBER_OF_STATES, NUMBER_OF_CONTROLS, Nn);
        mpc->x0 = x0;
        mpc->xf = xf;
        mpc->state_objective << POSITION_ERROR_COST_WEIGHT, 0.0, 0.0, VELOCITY_ERROR_COST_WEIGHT;
        mpc->control_objective << FORCE_COST_WEIGHT;
        mpc->state_dynamics << 1.0, (HORIZON_TIME / Nn), 0.0, 1.0;
        mpc->control_dynamics << 0.0, (HORIZON_TIME / Nn) / MASS;
        mpc->x_min << MIN_POSITION, MIN_VELOCITY;
        mpc->x_max << MAX_POSITION, MAX_VELOCITY;
        mpc->u_min << MIN_FORCE;
        mpc->u_max << MAX_FORCE;

        auto qp = mpc2qp(mpc);
        auto osqp = qp2osqp(qp);
        osqp->settings->verbose = false;
        osqp->settings->warm_starting = true;
        osqp->settings->polishing = true;

        const auto cstart = std::chrono::high_resolution_clock::now();
        for (int k = 1; k <= NUMBER_OF_MPC_ITERATIONS; k++)
        {

            solve_osqp(osqp);

            const auto qp_solution = get_solution(osqp);
            const auto mpc_solution = get_mpc_solution(NUMBER_OF_STATES, NUMBER_OF_CONTROLS, Nn, qp_solution);

            const float rand_float = (float)(rand()) / (float)(RAND_MAX);
            const float rand_force = 0.5F * (rand_float - 0.5F);

            const Float u0 = mpc_solution->ustar(0, 0);
            x0 = mpc->state_dynamics * x0 + mpc->control_dynamics * (u0 + rand_force);

            update_initial_state(qp, NUMBER_OF_STATES, x0);
            update_data(osqp, qp);
        }
        const auto cend = std::chrono::high_resolution_clock::now();

        const time_t duration = std::chrono::duration_cast<std::chrono::microseconds>(cend - cstart).count();
        const double kHz = (double)(NUMBER_OF_MPC_ITERATIONS * 1E3) / (double)(duration);

        std::cout << "Nodes: " << Nn << "\n";
        std::cout << "f = " << kHz << " kHz\n";
        std::cout << "x0: [ " << x0(0) << "," << x0(1) << " ]\n\n";

        out_file << kHz << ",";
        out_file << x0(0) << "," << x0(1);

        out_file << "\n";
    }

    out_file.close();

    return 0;
}