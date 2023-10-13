#ifndef ORLQP_QP_PROBLEM_HPP_
#define ORLQP_QP_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "QPModel.hpp"
#include "QPSolver.hpp"
#include "OSQPSolver.hpp"

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
            if (!qp_solution_)
                solveQP();
            return *qp_solution_;
        }

    protected:
        const std::shared_ptr<ModelType> model_;
        const std::shared_ptr<SolverType> solver_;

        std::shared_ptr<QPSolution> qp_solution_;

        void solveQP()
        {
            model_->setup();
            solver_->setup(*model_);
            if (solver_->solve(*model_))
                qp_solution_ = std::make_shared<QPSolution>(solver_->getSolution());
        }
    };

}

#endif