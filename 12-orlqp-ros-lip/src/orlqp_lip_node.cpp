#include "orlqp/orlqp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#define NODE_NAME "orlqp_lip_node"
#define SUBSCRIBE_TOPIC "/lip/info"
#define GOAL_TOPIC "/lip/goal"
#define PUBLISH_TOPIC "/lip/cmd"
#define PUBLISH_PERIOD 20ms

#define NUM_STATES 4
#define NUM_CONTROLS 1
#define NUM_NODES 25
#define TIME_HORIZON 5.0F
#define DELTA_TIME TIME_HORIZON / NUM_NODES

#define MIN_X -10.0
#define MAX_X 10.0
#define MIN_V -10.0
#define MAX_V 10.0
#define MIN_Q -10.0
#define MAX_Q 10.0
#define MIN_W -10.0
#define MAX_W 10.0
#define MIN_U -10.0
#define MAX_U 10.0

#define GRAVITY 9.81
#define CART_MASS 0.5
#define PEND_MASS 0.5
#define CART_FRIC 0.1
#define PEND_COML 0.3
#define PEND_INER 0.006

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

    const double A_div = PEND_INER * (CART_MASS + PEND_MASS) + CART_MASS * PEND_MASS * PEND_COML * PEND_COML;
    const double A_22n = -(PEND_INER + PEND_MASS * PEND_COML * PEND_COML) * CART_FRIC;
    const double A_23n = PEND_MASS * PEND_MASS * GRAVITY * PEND_COML * PEND_COML;
    const double A_42n = -PEND_MASS * PEND_COML * CART_FRIC;
    const double A_43n = PEND_MASS * GRAVITY * PEND_COML * (CART_MASS + PEND_MASS);
    const double B_21n = PEND_INER + PEND_MASS * PEND_COML * PEND_COML;
    const double B_41n = PEND_MASS * PEND_COML;

    // Set up MPC
    using namespace orlqp;
    MPCProblem::Ptr MPC = std::make_shared<MPCProblem>(NUM_STATES, NUM_CONTROLS, NUM_NODES);
    MPC->x0 << 0.0, 0.0, 0.0, 0.0;
    MPC->xf << 0.0, 0.0, 0.0, 0.0;
    MPC->state_objective << 10.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 10.0, 0.0,
        0.0, 0.0, 0.0, 1.0;
    MPC->control_objective << 1.0;
    MPC->state_dynamics << 1.0, DELTA_TIME, 0.0, 0.0,
        0.0, 1.0 + DELTA_TIME * A_22n / A_div, DELTA_TIME * A_23n / A_div, 0.0,
        0.0, 0.0, 1.0, DELTA_TIME,
        0.0, DELTA_TIME * A_42n / A_div, DELTA_TIME * A_43n / A_div, 1.0;
    MPC->control_dynamics << 0.0, DELTA_TIME * B_21n / A_div, 0.0, DELTA_TIME * B_41n / A_div;
    MPC->x_min << MIN_X, MIN_V, MIN_Q, MIN_W;
    MPC->x_max << MAX_X, MAX_V, MAX_Q, MAX_W;
    MPC->u_min << MIN_U;
    MPC->u_max << MAX_U;

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