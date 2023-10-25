#include "orlqp/orlqp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#define NODE_NAME "orlqp_di2d_array_node"
#define SUBSCRIBE_TOPIC "/di2d_array/info"
#define GOAL_TOPIC "/di2d_array/goal"
#define PUBLISH_TOPIC "/di2d_array/cmd"
#define PUBLISH_PERIOD 20ms

#define NUM_PROBLEMS 20
#define NUM_STATES 4
#define NUM_CONTROLS 2
#define NUM_NODES 15
#define TIME_HORIZON 1.0F
#define DELTA_TIME TIME_HORIZON / NUM_NODES

#define MIN_XY -10.0
#define MAX_XY 10.0
#define MIN_VXY -10.0
#define MAX_VXY 10.0
#define MIN_AXY -10.0
#define MAX_AXY 10.0

inline std::string vec_to_string(const std::vector<orlqp::Float> &vec)
{
    std::string str = " ";
    for (const auto val : vec)
        str += std::to_string(val) + " ";
    return str;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // Set up MPCs
    using namespace orlqp;
    std::vector<MPCProblem::Ptr> MPCs(NUM_PROBLEMS);
    std::vector<QPProblem::Ptr> QPs(NUM_PROBLEMS);
    for (int p = 0; p < NUM_PROBLEMS; p++)
    {
        // Set up MPC
        MPCProblem::Ptr MPC = std::make_shared<MPCProblem>(NUM_STATES, NUM_CONTROLS, NUM_NODES);
        MPC->x0 << 0.0, 0.0, 0.0, 0.0;
        MPC->xf << 0.0, 0.0, 0.0, 0.0;
        MPC->state_objective << 10.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 10.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
        MPC->control_objective << 1.0, 0.0,
            0.0, 1.0;
        MPC->state_dynamics << 1.0, DELTA_TIME, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, DELTA_TIME,
            0.0, 0.0, 0.0, 1.0;
        MPC->control_dynamics << 0.0, 0.0,
            DELTA_TIME, 0.0,
            0.0, 0.0,
            0.0, DELTA_TIME;
        MPC->x_min << MIN_XY, MIN_VXY, MIN_XY, MIN_VXY;
        MPC->x_max << MAX_XY, MAX_VXY, MAX_XY, MAX_VXY;
        MPC->u_min << MIN_AXY, MIN_AXY;
        MPC->u_max << MAX_AXY, MAX_AXY;

        MPCs[p] = MPC;
        QPs[p] = MPC->getQP();
    }

    // Set up QPArray
    QPArrayProblem::Ptr QP_array = std::make_shared<QPArrayProblem>(QPs);

    // Set up OSQP
    OSQP::Ptr osqp = std::make_shared<OSQP>();
    osqp->getSettings()->verbose = false;
    osqp->getSettings()->warm_starting = false;
    osqp->getSettings()->polishing = false;
    osqp->setup(QP_array->getQP());

    // Set up ROS node
    using namespace rclcpp;
    Node::SharedPtr node = std::make_shared<Node>(NODE_NAME);
    RCLCPP_INFO(node->get_logger(), "Created ROS node '%s'", NODE_NAME);

    // Set up state subscriber
    using namespace std_msgs::msg;
    Float64MultiArray::ConstSharedPtr latest_msg = nullptr;
    bool update_osqp = false;
    auto joint_sub = node->create_subscription<Float64MultiArray>(
        SUBSCRIBE_TOPIC, 10,
        [&](Float64MultiArray::ConstSharedPtr msg)
        {
            const std::vector<Float> state = msg->data;

            if (state.size() != NUM_PROBLEMS * NUM_STATES)
            {
                RCLCPP_ERROR(node->get_logger(), "Bad state message: [%s]", vec_to_string(state).c_str());
                return;
            }

            for (int p = 0; p < NUM_PROBLEMS; p++)
            {
                const int idx = p * NUM_STATES;
                const Eigen::Map<const EigenVector> x0_new(state.data() + idx, NUM_STATES);
                MPCs[p]->setInitialState(x0_new);
            }

            latest_msg = msg;
            update_osqp = true;
        });
    RCLCPP_INFO(node->get_logger(), "Subscribing to JointState topic '%s'", SUBSCRIBE_TOPIC);

    // Set up goal subscriber
    auto goal_sub = node->create_subscription<Float64MultiArray>(
        GOAL_TOPIC, 10,
        [&](Float64MultiArray::ConstSharedPtr msg)
        {
            const std::vector<double> goal = msg->data;

            if (goal.size() != NUM_PROBLEMS * NUM_STATES)
            {
                RCLCPP_ERROR(node->get_logger(), "Bad goal message: [%s]", vec_to_string(goal).c_str());
                return;
            }

            for (int p = 0; p < NUM_PROBLEMS; p++)
            {
                const int idx = p * NUM_STATES;
                const Eigen::Map<const EigenVector> xf_new(goal.data() + idx, NUM_STATES);
                MPCs[p]->setDesiredState(xf_new);
            }

            RCLCPP_INFO(node->get_logger(), "Updated goal state.");

            update_osqp = true;
        });

    // Set up command publisher
    using namespace std::chrono_literals;
    auto cmd_pub = node->create_publisher<Float64MultiArray>(PUBLISH_TOPIC, 10);
    RCLCPP_INFO(node->get_logger(), "Publishing to JointState topic '%s'", PUBLISH_TOPIC);

    // Run MPC
    using namespace std::chrono;
    Float64MultiArray::SharedPtr latest_cmd = std::make_shared<Float64MultiArray>();
    auto mpc_timer = node->create_wall_timer(
        PUBLISH_PERIOD,
        [&]()
        {
            if (!latest_msg)
                return;

            if (update_osqp)
            {
                QP_array->update();
                osqp->update();
                update_osqp = false;
            }

            osqp->solve();

            const QPSolution::Ptr qp_sol = osqp->getQPSolution();
            const std::vector<QPSolution::Ptr> qp_array_sol = QP_array->splitQPSolution(qp_sol);

            std::vector<Float> u_star(NUM_PROBLEMS * NUM_CONTROLS);
            for (int p = 0; p < NUM_PROBLEMS; p++)
            {
                const QPSolution::Ptr qp_sol_i = qp_array_sol[p];
                const MPCSolution::Ptr mpc_sol_i = MPCs[p]->getMPCSolution(qp_sol_i);

                const Float *u_data_i = mpc_sol_i->ustar.data();

                const int idx = p*NUM_CONTROLS;
                std::copy_n(u_data_i, NUM_CONTROLS, u_star.begin() + idx);
            }

            RCLCPP_INFO(node->get_logger(), "Publishing control.");

            latest_cmd->data = u_star;

            cmd_pub->publish(*latest_cmd);
            latest_msg = nullptr;
        });
    RCLCPP_INFO(node->get_logger(), "Running MPC with a period of %lu ms", duration_cast<milliseconds>(PUBLISH_PERIOD).count());

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}