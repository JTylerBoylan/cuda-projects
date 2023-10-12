#ifndef ORLQP_MPC_PROBLEM_HPP_
#define ORLQP_MPC_PROBLEM_HPP_

#include "QPTypes.hpp"
#include "QPProblem.hpp"

namespace boylan
{

    struct MPCSolution : public QPSolution
    {
        EigenVector u_star;
    };

    class MPCProblem : public QPProblem
    {
    public:
        /*
            MPC Problem:
            minimize x'*Q*x + u'*R*u
            subject to x(k+1) = A*x(k) + B*u(k)
                       x_min <= x <= x_max
                       u_min <= u <= u_max

            Q : State objective
            R : Control objective
            A : State dynamics
            B : Control dynamics
        */

        size_t &nodeCount()
        {
            return num_nodes_;
        }

        size_t &stateSize()
        {
            return num_states_;
        }

        size_t &controlSize()
        {
            return num_controls_;
        }

        EigenVector &getInitialState()
        {
            return initial_state_;
        }

        EigenVector &getDesiredState()
        {
            return desired_state_;
        }

        EigenMatrix &getStateObjective()
        {
            return state_objective_;
        }

        EigenMatrix &getControlObjective()
        {
            return control_objective_;
        }

        EigenMatrix &getStateDynamics()
        {
            return state_dynamics_;
        }

        EigenMatrix &getControlDynamics()
        {
            return control_dynamics_;
        }

        EigenVector &getStateLowerBound()
        {
            return state_lower_bound_;
        }

        EigenVector &getStateUpperBound()
        {
            return state_upper_bound_;
        }

        EigenVector &getControlLowerBound()
        {
            return control_lower_bound_;
        }

        EigenVector &getControlUpperBound()
        {
            return control_upper_bound_;
        }

        // updateInitialState(const EigenVector &x0) {}

        void setup() override
        {
            countVariables();
            countConstraints();
            calculateHessianMatrix();
            calculateGradientVector();
            calculateLinearConstraintMatrix();
            calculateBoundVectors();
            QPProblem::setup();
        }

        MPCSolution QPtoMPCSolution(const QPSolution &qp_solution);

    protected:
        size_t num_nodes_;
        size_t num_states_;
        size_t num_controls_;

        EigenVector initial_state_;
        EigenVector desired_state_;
        EigenMatrix state_objective_;
        EigenMatrix control_objective_;
        EigenMatrix state_dynamics_;
        EigenMatrix control_dynamics_;
        EigenVector state_lower_bound_;
        EigenVector state_upper_bound_;
        EigenVector control_lower_bound_;
        EigenVector control_upper_bound_;

        void countVariables();
        void countConstraints();
        void calculateHessianMatrix();
        void calculateGradientVector();
        void calculateLinearConstraintMatrix();
        void calculateBoundVectors();
    };

}

#endif