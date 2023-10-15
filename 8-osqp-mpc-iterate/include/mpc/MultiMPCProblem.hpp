#ifndef ORLQP_MULTI_MPC_PROBLEM_HPP_
#define ORLQP_MULTI_MPC_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "MultiQPProblem.hpp"
#include "mpc/MPCModel.hpp"

namespace boylan
{

    template <typename ModelType, typename SolverType = OSQP>
    class MultiMPCProblem : public MultiQPProblem<ModelType, SolverType>
    {
        static_assert(std::is_base_of<MPCModel, ModelType>::value, "ModelType must derive from MPCModel.");

    public:
        MultiMPCProblem(const int N)
            : MultiQPProblem<ModelType, SolverType>(N)
        {
            updated_.resize(N);
            for (int i = 0; i < N; i++)
                updated_[i] = true;
            mpc_solutions_.resize(N);
        }

        virtual void setup() override
        {
            MultiQPProblem<ModelType, SolverType>::setup();
        }

        void updateInitialState(const int index, const EigenVector &x0)
        {
            updated_[index] = true;
            const auto model = this->models_[index];
            const auto solver = this->solvers_[index];
            size_t size = model->getStateSize();
            model->getInitialState() = x0;
            model->getLowerBoundVector().block(0, 0, size, 1) = -x0;
            model->getUpperBoundVector().block(0, 0, size, 1) = -x0;
            solver->updateLowerBound(model->getLowerBoundVector());
            solver->updateUpperBound(model->getUpperBoundVector());
        }

        void updateDesiredState(const int index, const EigenVector &xf)
        {
            updated_[index] = true;
            const auto model = this->models_[index];
            const auto solver = this->solvers_[index];
            model->getDesiredState() = xf;
            model->calculateGradientVector();
            solver->updateGradient(model->getGradientVector());
        }

        MPCSolution &getSolution(const int index)
        {
            return *(mpc_solutions_[index]);
        }

        virtual void solve() override
        {
            std::for_each(std::execution::par, this->idx_.cbegin(), this->idx_.cend(), [&](const int i)
                          { if (updated_[i]) {
                                if (this->solvers_[i]->solve(*(this->models_[i])))
                                    this->qp_solutions_[i] = std::make_shared<QPSolution>(this->solvers_[i]->getSolution());
                                this->updateMPCSolution(i);
                                updated_[i] = false;
                            } });
        }

    protected:
        std::vector<bool> updated_;
        std::vector<std::shared_ptr<MPCSolution>> mpc_solutions_;

        void updateMPCSolution(const int index)
        {
            size_t nn = this->models_[index]->getNodeCount();
            size_t nx = this->models_[index]->getStateSize();
            size_t nu = this->models_[index]->getControlSize();

            auto mpc_solution = std::make_shared<MPCSolution>();
            auto qp_solution = this->qp_solutions_[index];
            mpc_solution->x_star = EigenVector(nx * (nn + 1));
            mpc_solution->x_star = qp_solution->x_star.block(0, 0, nx * (nn + 1), 1);
            mpc_solution->u_star = EigenVector(nu * nn);
            mpc_solution->u_star = qp_solution->x_star.block(nx * (nn + 1), 0, nu * nn, 1);
            mpc_solution->run_time_s = qp_solution->run_time_s;
            mpc_solution->setup_time_s = qp_solution->setup_time_s;
            mpc_solution->solve_time_s = qp_solution->solve_time_s;

            mpc_solutions_[index] = mpc_solution;
        }
    };

}

#endif