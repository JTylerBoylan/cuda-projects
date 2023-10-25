#ifndef ORLQP_MPC_PROBLEM_HPP_
#define ORLQP_MPC_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "QPProblem.hpp"
#include "mpc/MPCModel.hpp"

namespace boylan
{

    template <typename ModelType, typename SolverType = OSQP>
    class MPCProblem : public QPProblem<ModelType, SolverType>
    {
        static_assert(std::is_base_of<MPCModel, ModelType>::value, "ModelType must derive from MPCModel.");

    public:
        MPCProblem()
            : QPProblem<ModelType, SolverType>()
        {
        }

        virtual void setup()
        {
            QPProblem<ModelType, SolverType>::setup();
        }

        int updateInitialState(const EigenVector &x0)
        {
            updated_ = true;
            size_t size = this->model_->getStateSize();
            this->model_->getInitialState() = x0;
            this->model_->getLowerBoundVector().block(0, 0, size, 1) = -x0;
            this->model_->getUpperBoundVector().block(0, 0, size, 1) = -x0;
            int exlow = this->solver_->updateLowerBound(this->model_->getLowerBoundVector());
            int exup = this->solver_->updateUpperBound(this->model_->getUpperBoundVector());
            return std::max(exlow, exup);
        }

        int updateDesiredState(const EigenVector &xf)
        {
            updated_ = true;
            this->model_->getDesiredState() = xf;
            this->model_->calculateGradientVector();
            return this->solver_->updateGradient(this->model_->getGradientVector());
        }

        MPCSolution &getSolution()
        {
            return *mpc_solution_;
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
        bool updated_ = true;
        std::shared_ptr<MPCSolution> mpc_solution_;

        void updateMPCSolution()
        {
            size_t N = this->model_->getNodeCount();
            size_t nx = this->model_->getStateSize();
            size_t nu = this->model_->getControlSize();

            mpc_solution_ = std::make_shared<MPCSolution>();

            mpc_solution_->x_star = EigenVector(nx * (N + 1));
            mpc_solution_->x_star = this->qp_solution_->x_star.block(0, 0, nx * (N + 1), 1);
            mpc_solution_->u_star = EigenVector(nu * N);
            mpc_solution_->u_star = this->qp_solution_->x_star.block(nx * (N + 1), 0, nu * N, 1);
            mpc_solution_->run_time_s = this->qp_solution_->run_time_s;
            mpc_solution_->setup_time_s = this->qp_solution_->setup_time_s;
            mpc_solution_->solve_time_s = this->qp_solution_->solve_time_s;
        }
    };

}

#endif