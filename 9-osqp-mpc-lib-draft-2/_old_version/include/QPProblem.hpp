#ifndef ORLQP_QP_PROBLEM_HPP_
#define ORLQP_QP_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "QPModel.hpp"
#include "QPSolver.hpp"
#include "osqp/OSQPSolver.hpp"

namespace boylan
{

    template <typename ModelType, typename SolverType = OSQP>
    class QPProblem
    {
        static_assert(std::is_base_of<QPModel, ModelType>::value, "ModelType must derive from QPModel.");
        static_assert(std::is_base_of<QPSolver, SolverType>::value, "SolverType must derive from QPSolver.");

    public:
        QPProblem()
            : model_(std::make_shared<ModelType>()), solver_(std::make_shared<SolverType>())
        {
        }

        virtual void setup()
        {
            model_->setup(0);
            solver_->setup(*model_);
        }

        const std::shared_ptr<ModelType> getModel()
        {
            return model_;
        }

        const std::shared_ptr<SolverType> getSolver()
        {
            return solver_;
        }

        QPSolution &getSolution()
        {
            return *qp_solution_;
        }

        virtual void solve()
        {
            if (solver_->solve(*model_))
                qp_solution_ = std::make_shared<QPSolution>(solver_->getSolution());
        }

    protected:
        std::shared_ptr<ModelType> model_;
        std::shared_ptr<SolverType> solver_;

        std::shared_ptr<QPSolution> qp_solution_;
    };

}

#endif