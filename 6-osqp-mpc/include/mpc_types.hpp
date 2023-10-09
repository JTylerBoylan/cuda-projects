#ifndef ONR_OSQP_MPC_TYPES_HPP_
#define ONR_OSQP_MPC_TYPES_HPP_

#include <Eigen/Dense>

namespace boylan
{

    // a = number of states
    // b = number of controls
    // n = number of decision variables
    // m = number of constraints

    using namespace Eigen;

    typedef struct
    {
        MatrixXf A; // [axa]
        MatrixXf B; // [axb]
        // s.t x(k+1) = A * x(k) + B * u(k)
    } MPCDynamics;

    typedef struct
    {
        VectorXf x_min, x_max; // [ax1]
        VectorXf u_min, u_max; // [bx1]
        // s.t x_min <= x(k) <= x_max
        // and u_min <= u(k) <= u_max
    } MPCConstraints;

    typedef struct
    {
        MatrixXf Q; // [nxn]
        VectorXf R; // [nx1]
        // s.t f(x) = x' * Q * x + R' * x
    } MPCObjective;

    typedef struct
    {
        MPCDynamics dynamics;
        MPCConstraints constraints;
        MPCObjective objective;
    } MPCProblem;

    typedef struct
    {
        VectorXf x_star;
    } MPCResult;

}

#endif