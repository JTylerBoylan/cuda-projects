#ifndef ORLQP_MPC_PROBLEM_HPP_
#define ORLQP_MPC_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "QPProblem.hpp"
#include "mpc/MPCArrayModel.hpp"

namespace boylan
{

    template <typename ModelType, typename SolverType = OSQP>
    class MPCArrayProblem : public QPProblem<ModelType, SolverType>
    {
        static_assert(std::is_base_of<MPCArrayModel, ModelType>::value, "ModelType must derive from MPCArrayModel.");

    public:
        MPCArrayProblem(const int num_problems)
            : QPProblem<ModelType, SolverType>(), num_problems_(num_problems)
        {
            this->model_ = std::make_shared<ModelType>(num_problems);
        }

        virtual void setup()
        {
            QPProblem<ModelType, SolverType>::setup();
        }

        void updateInitialState(const int index, const EigenVector &x0)
        {
            const int start_idx = this->model_->getVariableStartIndex(index);
            updated_ = true;
            size_t size = this->model_->getStateSize(index);
            this->model_->getInitialState(index) = x0;
            this->model_->getLowerBoundVector().block(start_idx + 0, 0, size, 1) = -x0;
            this->model_->getUpperBoundVector().block(start_idx + 0, 0, size, 1) = -x0;
            this->solver_->updateLowerBound(this->model_->getLowerBoundVector());
            this->solver_->updateUpperBound(this->model_->getUpperBoundVector());
        }

        void updateDesiredState(const int index, const EigenVector &xf)
        {
            updated_ = true;
            this->model_->getDesiredState(index) = xf;
            this->model_->calculateGradientVector();
            this->solver_->updateGradient(this->model_->getGradientVector());
        }

        MPCArraySolution &getSolution()
        {
            return *(mpc_solution_);
        }

        virtual void solve() override
        {
            if (updated_)
            {
                QPProblem<ModelType, SolverType>::solve();
                this->updateMPCSolution();
                updated_ = false;
            }
        }

    protected:
        const size_t num_problems_;

        bool updated_ = true;
        std::shared_ptr<MPCArraySolution> mpc_solution_;

        void updateMPCSolution()
        {
            const auto qp_solution = this->qp_solution_;
            auto mpc_solution = std::make_shared<MPCArraySolution>();

            mpc_solution->run_time_s = qp_solution->run_time_s;
            mpc_solution->setup_time_s = qp_solution->setup_time_s;
            mpc_solution->solve_time_s = qp_solution->solve_time_s;
            mpc_solution->x_star.resize(num_problems_);
            mpc_solution->u_star.resize(num_problems_);

            size_t start_idx = 0;
            for (int p = 0; p < num_problems_; p++)
            {
                const size_t N = this->model_->getNodeCount(p);
                const size_t nx = this->model_->getStateSize(p);
                const size_t nu = this->model_->getControlSize(p);

                mpc_solution->x_star[p] = EigenVector(nx * (N + 1));
                mpc_solution->x_star[p] = qp_solution->x_star.block(start_idx, 0, nx * (N + 1), 1);
                mpc_solution->u_star[p] = EigenVector(nu * N);
                mpc_solution->u_star[p] = qp_solution->x_star.block(start_idx + nx * (N + 1), 0, nu * N, 1);
                
                start_idx += nx * (N + 1) + nu * N;
            }

            mpc_solution_ = mpc_solution;
        }
    };

}

#endif