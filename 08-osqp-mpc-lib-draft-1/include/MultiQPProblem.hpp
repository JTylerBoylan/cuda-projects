#ifndef ORLQP_MULTI_QP_PROBLEM_HPP_
#define ORLQP_MULTI_QP_PROBLEM_HPP_

#include <algorithm>
#include <execution>

#include "QPTypes.hpp"
#include "QPModel.hpp"
#include "QPSolver.hpp"
#include "osqp/OSQPSolver.hpp"

namespace boylan
{

    template <typename ModelType, typename SolverType = OSQP>
    class MultiQPProblem
    {
        static_assert(std::is_base_of<QPModel, ModelType>::value, "ModelType must derive from QPModel.");
        static_assert(std::is_base_of<QPSolver, SolverType>::value, "SolverType must derive from QPSolver.");

    public:
        MultiQPProblem(const int N)
            : N_(N)
        {
            models_.resize(N);
            solvers_.resize(N);
            qp_solutions_.resize(N);
            idx_.resize(N);
            std::iota(idx_.begin(), idx_.end(), 0);
            for (int i = 0; i < N; i++)
            {
                models_[i] = std::make_shared<ModelType>();
                solvers_[i] = std::make_shared<SolverType>();
            }
        }

        virtual void setup()
        {
            for (int i = 0; i < N_; i++)
            {
                models_[i]->setup(i);
                solvers_[i]->setup(*(models_[i]));
            }
        }

        const size_t getSize() const
        {
            return N_;
        }

        std::shared_ptr<ModelType> getModel(const int index)
        {
            return models_[index];
        }

        std::shared_ptr<SolverType> getSolver(const int index)
        {
            return solvers_[index];
        }

        std::vector<int>::const_iterator beginIndex()
        {
            return idx_.cbegin();
        }

        std::vector<int>::const_iterator endIndex()
        {
            return idx_.cend();
        }

        QPSolution &getSolution(const int index)
        {
            return *(qp_solutions_[index]);
        }

        virtual void solve()
        {
            std::for_each(std::execution::par, idx_.cbegin(), idx_.cend(), [&](const int i)
                          { if (solvers_[i]->solve(*(models_[i])))
                                qp_solutions_[i] = std::make_shared<QPSolution>(solvers_[i]->getSolution()); });
        }

    protected:
        const int N_;

        std::vector<std::shared_ptr<ModelType>> models_;
        std::vector<std::shared_ptr<SolverType>> solvers_;

        std::vector<std::shared_ptr<QPSolution>> qp_solutions_;

        std::vector<int> idx_;
    };

}

#endif