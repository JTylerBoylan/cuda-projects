#include <iostream>

#include "orlqp/mpc_util.hpp"
#include "orlqp/osqp_util.hpp"

#define NUMBER_OF_STATES 2
#define NUMBER_OF_CONTROLS 1

#define NUMBER_OF_NODES 11

#define POSITION_ERROR_COST_WEIGHT 1.0
#define VELOCITY_ERROR_COST_WEIGHT 0.0
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
    EigenVector x0(NUMBER_OF_STATES), xf(NUMBER_OF_STATES);
    x0 << 1.0, 0.0;
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

    auto qp = mpc2qp(mpc);

    auto osqp = qp2osqp(qp);

    osqp->settings->verbose = true;
    update_settings(osqp);
    
    solve_osqp(osqp);

    return 0;
}