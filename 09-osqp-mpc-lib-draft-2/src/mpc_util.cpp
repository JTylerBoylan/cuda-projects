#include "orlqp/mpc_util.hpp"

namespace orlqp
{

    MPCProblem::Ptr create_mpc(const int Nx, const int Nu, const int Nn)
    {
        return std::make_shared<MPCProblem>(Nx, Nu, Nn);
    }

    QPProblem::Ptr mpc2qp(const MPCProblem::Ptr mpc)
    {
        const int Nx = mpc->num_states;
        const int Nu = mpc->num_controls;
        const int Nn = mpc->num_nodes;
        const int n = Nx * (Nn + 1) + Nu * Nn;
        const int m = 2 * Nx * (Nn + 1) + Nu * Nn;
        QPProblem::Ptr qp = std::make_shared<QPProblem>(n, m);
        calculate_mpc2qp_hessian(n, qp->hessian, Nx, Nu, Nn, mpc->state_objective, mpc->control_objective);
        calculate_mpc2qp_gradient(n, qp->gradient, Nx, Nu, Nn, mpc->state_objective, mpc->xf);
        calculate_mpc2qp_linear_constraint(n, m, qp->linear_constraint, Nx, Nu, Nn, mpc->state_dynamics, mpc->control_dynamics);
        calculate_mpc2qp_lower_bound(m, qp->lower_bound, Nx, Nu, Nn, mpc->x0, mpc->x_min, mpc->x_max, mpc->u_min, mpc->u_max);
        calculate_mpc2qp_upper_bound(m, qp->upper_bound, Nx, Nu, Nn, mpc->x0, mpc->x_min, mpc->x_max, mpc->u_min, mpc->u_max);
        return qp;
    }

    MPCSolution::Ptr get_mpc_solution(const int Nx, const int Nu, const int Nn,
                                      const QPSolution::Ptr qp_solution)
    {
        auto mpc_solution = std::make_shared<MPCSolution>();
        mpc_solution->run_time_s = qp_solution->run_time_s;
        mpc_solution->setup_time_s = qp_solution->setup_time_s;
        mpc_solution->solve_time_s = qp_solution->solve_time_s;
        mpc_solution->xstar = qp_solution->xstar.block(0, 0, Nx * (Nn + 1), 1);
        mpc_solution->ustar = qp_solution->xstar.block(Nx * (Nn + 1), 0, Nu * Nn, 1);
        return mpc_solution;
    }

    void calculate_mpc2qp_hessian(const int n,
                    EigenSparseMatrix &H,
                    const int Nx, const int Nu, const int Nn,
                    const EigenMatrix &Q, const EigenMatrix &R)
    {
        std::vector<EigenTriplet> triplets;
        for (int i = 0; i < Nn + 1; i++)
        {
            const int idx = Nx * i;
            for (int j = 0; j < Nx; j++)
                for (int k = 0; k < Nx; k++)
                {
                    const Float value = Q(j, k);
                    if (value != 0)
                        triplets.push_back(EigenTriplet(idx + k, idx + k, value));
                }
        }
        for (int i = 0; i < Nn; i++)
        {
            const int idx = Nx * (Nn + 1) + Nu * i;
            for (int j = 0; j < Nu; j++)
                for (int k = 0; k < Nu; k++)
                {
                    const Float value = R(j, k);
                    if (value != 0)
                        triplets.push_back(EigenTriplet(idx + k, idx + k, value));
                }
        }
        H.setFromTriplets(triplets.begin(), triplets.end());
    }

    void calculate_mpc2qp_gradient(const int n,
                     EigenVector &G,
                     const int Nx, const int Nu, const int Nn,
                     const EigenMatrix &Q, const EigenVector xf)
    {
        const EigenVector Gx = Q * (-xf);
        for (int i = 0; i < n; i++)
        {
            if (i < Nx * (Nn + 1))
            {
                int gIdx = i % Nx;
                G(i, 0) = Gx(gIdx, 0);
            }
            else
            {
                G(i, 0) = 0;
            }
        }
    }

    void calculate_mpc2qp_linear_constraint(const int n, const int m,
                              EigenSparseMatrix &Ac,
                              const int Nx, const int Nu, const int Nn,
                              const EigenMatrix &A, const EigenMatrix &B)
    {
        std::vector<EigenTriplet> triplets;
        for (int i = 0; i < Nx * (Nn + 1); i++)
            triplets.push_back(EigenTriplet(i, i, -1));

        for (int i = 0; i < Nn; i++)
            for (int j = 0; j < Nx; j++)
                for (int k = 0; k < Nx; k++)
                {
                    const Float value = A(j, k);
                    if (value != 0)
                        triplets.push_back(EigenTriplet(
                            Nx * (i + 1) + j,
                            Nx * i + k,
                            value));
                }

        for (int i = 0; i < Nn; i++)
            for (int j = 0; j < Nx; j++)
                for (int k = 0; k < Nu; k++)
                {
                    const Float value = B(j, k);
                    if (value != 0)
                        triplets.push_back(EigenTriplet(
                            Nx * (i + 1) + j,
                            Nu * i + k + Nx * (Nn + 1),
                            value));
                }

        for (int i = 0; i < n; i++)
            triplets.push_back(EigenTriplet(i + (Nn + 1) * Nx, i, 1));

        Ac.setFromTriplets(triplets.begin(), triplets.end());
    }

    void calculate_mpc2qp_lower_bound(const int m,
                        EigenVector &lb,
                        const int Nx, const int Nu, const int Nn,
                        const EigenVector x0,
                        const EigenVector &x_min, const EigenVector &x_max,
                        const EigenVector &u_min, const EigenVector &u_max)
    {
        const int Neq = Nx * (Nn + 1);
        const int Nineq = Nx * (Nn + 1) + Nu * Nn;

        EigenVector lower_inequality = EigenVector::Zero(Nineq);
        for (int i = 0; i < Nn + 1; i++)
            lower_inequality.block(Nx * i, 0, Nx, 1) = x_min;
        for (int i = 0; i < Nn; i++)
            lower_inequality.block(Nu * i + Nx * (Nn + 1), 0, Nu, 1) = u_min;

        EigenVector lower_equality = EigenVector::Zero(Neq);
        lower_equality.block(0, 0, Nx, 1) = -x0;

        lb.block(0, 0, Neq, 1) = lower_equality;
        lb.block(Neq, 0, Nineq, 1) = lower_inequality;
    }

    void calculate_mpc2qp_upper_bound(const int m,
                        EigenVector &ub,
                        const int Nx, const int Nu, const int Nn,
                        const EigenVector x0,
                        const EigenVector &x_min, const EigenVector &x_max,
                        const EigenVector &u_min, const EigenVector &u_max)
    {
        const int Neq = Nx * (Nn + 1);
        const int Nineq = Nx * (Nn + 1) + Nu * Nn;

        EigenVector upper_inequality = EigenVector::Zero(Nineq);
        for (int i = 0; i < Nn + 1; i++)
            upper_inequality.block(Nx * i, 0, Nx, 1) = x_max;
        for (int i = 0; i < Nn; i++)
            upper_inequality.block(Nu * i + Nx * (Nn + 1), 0, Nu, 1) = u_max;

        EigenVector upper_equality = EigenVector::Zero(Neq);
        upper_equality.block(0, 0, Nx, 1) = -x0;

        ub.block(0, 0, Neq, 1) = upper_equality;
        ub.block(Neq, 0, Nineq, 1) = upper_inequality;
    }

    void update_initial_state(QPProblem::Ptr qp, const int Nx, const EigenVector &x0)
    {
        qp->lower_bound.block(0, 0, Nx, 1) = -x0;
        qp->upper_bound.block(0, 0, Nx, 1) = -x0;
        qp->update.lower_bound = true;
        qp->update.upper_bound = true;
    }
}