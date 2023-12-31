#include "orlqp/orlqp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#define NODE_NAME "orlqp_di2d_node"
#define SUBSCRIBE_TOPIC "/di2d/info"
#define GOAL_TOPIC "/di2d/goal"
#define PUBLISH_TOPIC "/di2d/cmd"
#define PUBLISH_PERIOD 20ms

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

    // Set up MPC
    using namespace orlqp;
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

    // Set up OSQP
    OSQP::Ptr osqp = std::make_shared<OSQP>();
    osqp->getSettings()->verbose = false;
    osqp->getSettings()->warm_starting = false;
    osqp->getSettings()->polishing = false;
    osqp->setup(MPC->getQP());

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
            const std::vector<double> state = msg->data;

            if (state.size() != NUM_STATES)
            {
                RCLCPP_ERROR(node->get_logger(), "Bad state message: [%s]", vec_to_string(state).c_str());
                return;
            }

            const Eigen::Map<const EigenVector> x0_new(state.data(), NUM_STATES);

            MPC->setInitialState(x0_new);

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

            if (goal.size() != NUM_STATES)
            {
                RCLCPP_ERROR(node->get_logger(), "Bad goal message: [%s]", vec_to_string(goal).c_str());
                return;
            }

            const Eigen::Map<const EigenVector> xf_new(goal.data(), NUM_STATES);

            MPC->setDesiredState(xf_new);

            RCLCPP_INFO(node->get_logger(), "Set goal state to: [%s]", vec_to_string(goal).c_str());

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
                osqp->update();
                update_osqp = false;
            }

            osqp->solve();

            const QPSolution::Ptr qp_sol = osqp->getQPSolution();
            const MPCSolution::Ptr mpc_sol = MPC->getMPCSolution(qp_sol);

            const Float *u_data = mpc_sol->ustar.data();
            const std::vector<Float> u_star(u_data, u_data + NUM_CONTROLS);

            RCLCPP_INFO(node->get_logger(), "Publishing control: [%s]", vec_to_string(u_star).c_str());

            latest_cmd->data = u_star;

            cmd_pub->publish(*latest_cmd);
            latest_msg = nullptr;
        });
    RCLCPP_INFO(node->get_logger(), "Running MPC with a period of %lu ms", duration_cast<milliseconds>(PUBLISH_PERIOD).count());

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}