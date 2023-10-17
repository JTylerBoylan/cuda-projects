#ifndef ORLQP_QP_ARRAY_UTIL_HPP_
#define ORLQP_QP_ARRAY_UTIL_HPP_

#include "orlqp/types.hpp"
#include "orlqp/qp_problem_array.hpp"

namespace orlqp 
{

    QPProblemArray::Ptr create_qp_array(const std::vector<QPProblem::Ptr> &qps);


    void to_hessian(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps);

    void to_gradient(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps);

    void to_linear_constraint(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps);

    void to_lower_bound(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps);

    void to_upper_bound(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps);


    void update_hessian(QPProblemArray::Ptr qp_array, const int index, const EigenSparseMatrix &hessian_i);

    void update_gradient(QPProblemArray::Ptr qp_array, const int index, const EigenVector &gradient_i);

    void update_linear_constraint(QPProblemArray::Ptr qp_array, const int index, const EigenSparseMatrix &linear_constraint_i);

    void update_lower_bound(QPProblemArray::Ptr qp_array, const int index, const EigenVector &lower_bound_i);

    void update_upper_bound(QPProblemArray::Ptr qp_array, const int index, const EigenVector &upper_bound_i);

}

#endif