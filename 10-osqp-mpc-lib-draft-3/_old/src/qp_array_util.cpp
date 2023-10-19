#include "orlqp/qp_array_util.hpp"

namespace orlqp
{

    QPProblemArray::Ptr create_qp_array(const std::vector<QPProblem::Ptr> &qps)
    {
        const int Np = qps.size();

        int n = 0, m = 0;
        std::vector<int> varmap(Np), conmap(Np);
        for (int p = 0; p < Np; p++)
        {
            varmap[p] = n;
            conmap[p] = m;
            n += qps[p]->num_variables;
            m += qps[p]->num_constraints;
        }

        QPProblemArray::Ptr qp_array = std::make_shared<QPProblemArray>(n, m, Np);
        qp_array->variable_idx_map = varmap;
        qp_array->constraint_idx_map = conmap;

        calculate_qp_array_hessian(qp_array, qps);
        calculate_qp_array_gradient(qp_array, qps);
        calculate_qp_array_linear_constraint(qp_array, qps);
        calculate_qp_array_lower_bound(qp_array, qps);
        calculate_qp_array_upper_bound(qp_array, qps);

        return qp_array;
    }

    void calculate_qp_array_hessian(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps)
    {
        std::vector<EigenTriplet> triplets;
        for (int p = 0; p < qp_array->num_problems; p++)
        {
            const int var_idx = qp_array->variable_idx_map[p];
            const EigenSparseMatrix &hessian_p = qps[p]->hessian;
            triplets.reserve(hessian_p.nonZeros());
            for (int k = 0; k < hessian_p.outerSize(); k++)
                for (EigenSparseMatrix::InnerIterator it(hessian_p, k); it; ++it)
                    triplets.push_back(EigenTriplet(var_idx + it.row(), var_idx + it.col(), it.value()));
        }
        qp_array->hessian.setFromTriplets(triplets.begin(), triplets.end());
    }

    void calculate_qp_array_gradient(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps)
    {
        for (int p = 0; p < qp_array->num_problems; p++)
        {
            const int var_idx = qp_array->variable_idx_map[p];
            qp_array->gradient.block(var_idx, 0, qps[p]->num_variables, 1) = qps[p]->gradient;
        }
    }

    void calculate_qp_array_linear_constraint(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps)
    {
        std::vector<EigenTriplet> triplets;
        for (int p = 0; p < qp_array->num_problems; p++)
        {
            const int var_idx = qp_array->variable_idx_map[p];
            const int con_idx = qp_array->constraint_idx_map[p];
            const EigenSparseMatrix &linear_constraint_p = qps[p]->linear_constraint;
            triplets.reserve(linear_constraint_p.nonZeros());
            for (int k = 0; k < linear_constraint_p.outerSize(); k++)
                for (EigenSparseMatrix::InnerIterator it(linear_constraint_p, k); it; ++it)
                    triplets.push_back(EigenTriplet(con_idx + it.row(), var_idx + it.col(), it.value()));
        }
        qp_array->linear_constraint.setFromTriplets(triplets.begin(), triplets.end());
    }

    void calculate_qp_array_lower_bound(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps)
    {
        for (int p = 0; p < qp_array->num_problems; p++)
        {
            const int con_idx = qp_array->constraint_idx_map[p];
            qp_array->lower_bound.block(con_idx, 0, qps[p]->num_constraints, 1) = qps[p]->lower_bound;
        }
    }

    void calculate_qp_array_upper_bound(QPProblemArray::Ptr qp_array, const std::vector<QPProblem::Ptr> &qps)
    {
        for (int p = 0; p < qp_array->num_problems; p++)
        {
            const int con_idx = qp_array->constraint_idx_map[p];
            qp_array->upper_bound.block(con_idx, 0, qps[p]->num_constraints, 1) = qps[p]->upper_bound;
        }
    }

    void update_qp_array_hessian(QPProblemArray::Ptr qp_array, const int index, const EigenSparseMatrix &hessian_i)
    {
        const int var_idx = qp_array->variable_idx_map[index];

        for (int k = 0; k < qp_array->hessian.outerSize(); ++k)
            for (EigenSparseMatrix::InnerIterator it(qp_array->hessian, k); it; ++it)
                if (it.row() >= var_idx && it.col() >= var_idx &&
                    it.row() < var_idx + hessian_i.rows() && it.col() < var_idx + hessian_i.cols())
                    it.valueRef() = 0;

        for (int k = 0; k < hessian_i.outerSize(); ++k)
            for (EigenSparseMatrix::InnerIterator it(hessian_i, k); it; ++it)
                qp_array->hessian.coeffRef(it.row() + var_idx, it.col() + var_idx) = it.value();

        qp_array->hessian.prune([](const int &, const int &, const double &value)
                                { return value != 0.0; });
        qp_array->update.hessian = true;
    }

    void update_qp_array_gradient(QPProblemArray::Ptr qp_array, const int index, const EigenVector &gradient_i)
    {
        const int var_idx = qp_array->variable_idx_map[index];
        qp_array->gradient.block(var_idx, 0, gradient_i.rows(), 1) = gradient_i;
        qp_array->update.gradient = true;
    }

    void update_qp_array_linear_constraint(QPProblemArray::Ptr qp_array, const int index, const EigenSparseMatrix &linear_constraint_i)
    {
        const int var_idx = qp_array->variable_idx_map[index];
        const int con_idx = qp_array->constraint_idx_map[index];
        for (int k = 0; k < qp_array->linear_constraint.outerSize(); ++k)
            for (EigenSparseMatrix::InnerIterator it(qp_array->linear_constraint, k); it; ++it)
                if (it.row() >= var_idx && it.col() >= var_idx &&
                    it.row() < var_idx + linear_constraint_i.rows() && it.col() < var_idx + linear_constraint_i.cols())
                    it.valueRef() = 0;

        for (int k = 0; k < linear_constraint_i.outerSize(); ++k)
            for (EigenSparseMatrix::InnerIterator it(linear_constraint_i, k); it; ++it)
                qp_array->linear_constraint.coeffRef(it.row() + var_idx, it.col() + var_idx) = it.value();

        qp_array->linear_constraint.prune([](const int &, const int &, const double &value)
                                          { return value != 0.0; });
        qp_array->update.linear_constraint = true;
    }

    void update_qp_array_lower_bound(QPProblemArray::Ptr qp_array, const int index, const EigenVector &lower_bound_i)
    {
        const int con_idx = qp_array->constraint_idx_map[index];
        qp_array->lower_bound.block(con_idx, 0, lower_bound_i.rows(), 1) = lower_bound_i;
        qp_array->update.lower_bound = true;
    }

    void update_qp_array_upper_bound(QPProblemArray::Ptr qp_array, const int index, const EigenVector &upper_bound_i)
    {
        const int con_idx = qp_array->constraint_idx_map[index];
        qp_array->upper_bound.block(con_idx, 0, upper_bound_i.rows(), 1) = upper_bound_i;
        qp_array->update.upper_bound = true;
    }

}
