#include <iostream>

#include "orlqp/mpc_util.hpp"
#include "orlqp/osqp_util.hpp"
#include "orlqp/qp_array_util.hpp"

#define NUMBER_OF_PROBLEMS 3

#define NUMBER_OF_STATES 2
#define NUMBER_OF_CONTROLS 1

#define NUMBER_OF_NODES 11

#define POSITION_ERROR_COST_WEIGHT 1.0
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

using namespace orlqp;

int main()
{

    std::vector<QPProblem::Ptr> qps(NUMBER_OF_PROBLEMS);
    for (int p = 0; p < NUMBER_OF_PROBLEMS; p++)
    {
        EigenVector x0(NUMBER_OF_STATES), xf(NUMBER_OF_STATES);
        x0 << -4.0, 0.0;
        xf << 0.0, 0.0;

        auto mpc = create_mpc(NUMBER_OF_STATES, NUMBER_OF_CONTROLS, NUMBER_OF_NODES);
        mpc->x0 = x0;
        mpc->xf = xf;
        mpc->state_objective << POSITION_ERROR_COST_WEIGHT, 0.0, 0.0, VELOCITY_ERROR_COST_WEIGHT;
        mpc->control_objective << FORCE_COST_WEIGHT;
        mpc->state_dynamics << 1.0, (HORIZON_TIME / NUMBER_OF_NODES), 0.0, 1.0;
        mpc->control_dynamics << 0.0, (HORIZON_TIME / NUMBER_OF_NODES) / MASS;
        mpc->x_min << MIN_POSITION, MIN_VELOCITY;
        mpc->x_max << MAX_POSITION, MAX_VELOCITY;
        mpc->u_min << MIN_FORCE;
        mpc->u_max << MAX_FORCE;

        qps[p] = mpc2qp(mpc);
    }

    auto qp_array = create_qp_array(qps);

    std::cout << qp_array->hessian << std::endl;

    auto osqp = qp2osqp(qp_array);

    osqp->settings->verbose = true;
    osqp->settings->warm_starting = true;
    osqp->settings->polishing = true;

    solve_osqp(osqp);

    auto qp_solution = get_solution(osqp);

    std::cout << "xstar:\n"
              << qp_solution->xstar << "\n";

    return 0;
}