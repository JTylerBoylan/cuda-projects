#include "orlqp/orlqp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#define NODE_NAME "orlqp_lip_node"
#define SUBSCRIBE_TOPIC "/lip/info"
#define PUBLISH_TOPIC "/lip/cmd"
#define PUBLISH_PERIOD 20ms

#define NUM_STATES 2
#define NUM_CONTROLS 1
#define NUM_NODES 15
#define TIME_HORIZON 1.0F

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    // Set up MPC
    using namespace orlqp;
    MPCProblem::Ptr MPC = std::make_shared<MPCProblem>(NUM_STATES, NUM_CONTROLS, NUM_NODES);
    /**
     *
     * TODO: Setup MPC
     *
     */

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
    using namespace sensor_msgs::msg;
    JointState::ConstSharedPtr latest_msg = nullptr;
    auto joint_sub = node->create_subscription<JointState>(
        SUBSCRIBE_TOPIC, 10,
        [&](JointState::ConstSharedPtr msg)
        {
            /**
             *
             * TODO: Convert JointState position/velocity to x0
             *
             */
            latest_msg = msg;
        });
    RCLCPP_INFO(node->get_logger(), "Subscribing to JointState topic '%s'", SUBSCRIBE_TOPIC);

    // Set up command publisher
    using namespace std::chrono_literals;
    JointState::SharedPtr latest_cmd = std::make_shared<JointState>();
    auto cmd_pub = node->create_publisher<JointState>(PUBLISH_TOPIC, 10);
    RCLCPP_INFO(node->get_logger(), "Publishing to JointState topic '%s'", PUBLISH_TOPIC);

    // Run MPC
    using namespace std::chrono;
    auto mpc_timer = node->create_wall_timer(
        PUBLISH_PERIOD,
        [&]()
        {
            if (!latest_msg)
                return;

            osqp->update();
            osqp->solve();

            QPSolution::Ptr qp_sol = osqp->getQPSolution();
            MPCSolution::Ptr mpc_sol = MPC->getMPCSolution(qp_sol);

            /**
             * 
             * TODO: Convert u_star to JointState effort
             * 
            */

            cmd_pub->publish(*latest_cmd);
            latest_msg = nullptr;
        });
    RCLCPP_INFO(node->get_logger(), "Running MPC with a period of %lu ms", duration_cast<milliseconds>(PUBLISH_PERIOD).count());

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}