#ifndef ONR_OSQP_TYPES_HPP_
#define ONR_OSQP_TYPES_HPP_

#include <memory>

#include "Eigen/Dense"

#include "osqp/osqp_api_types.h"

namespace boylan
{

#ifdef OSQP_USE_FLOAT
    using EigenVector = Eigen::VectorXf;
    using EigenMatrix = Eigen::MatrixXf;
#else
    using EigenVector = Eigen::VectorXd;
    using EigenMatrix = Eigen::MatrixXd;
#endif

}

#endif