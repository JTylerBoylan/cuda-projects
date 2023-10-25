#include "orlqp/orlqp.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#define NODE_NAME "orlqp_lip_node"
#define SUBSCRIBE_TOPIC "/lip/info"
#define GOAL_TOPIC "/lip/goal"
#define PUBLISH_TOPIC "/lip/cmd"
#define PUBLISH_PERIOD 20ms

#define NUM_STATES 4
#define NUM_CONTROLS 2
#define NUM_NODES 15
#define TIME_HORIZON 1.0F
#define DELTA_TIME TIME_HORIZON / NUM_NODES

#define MIN_XY -5.0
#define MAX_XY 5.0
#define MIN_VXY -5.0
#define MAX_VXY 5.0
#define MIN_AXY -5.0
#define MAX_AXY 5.0

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
    using namespace sensor_msgs::msg;
    JointState::ConstSharedPtr latest_msg = nullptr;
    bool update_osqp = false;
    auto joint_sub = node->create_subscription<JointState>(
        SUBSCRIBE_TOPIC, 10,
        [&](JointState::ConstSharedPtr msg)
        {
            const std::vector<double> pos = msg->position;
            const std::vector<double> vel = msg->velocity;

            if (pos.size() != 2 || vel.size() != 2)
                return;

            EigenVector x0_new(NUM_STATES);
            x0_new << pos[0], vel[0], pos[1], vel[1];

            MPC->setInitialState(x0_new);

            latest_msg = msg;
            update_osqp = true;
        });
    RCLCPP_INFO(node->get_logger(), "Subscribing to JointState topic '%s'", SUBSCRIBE_TOPIC);

    // Set up goal subscriber
    auto goal_sub = node->create_subscription<JointState>(
        GOAL_TOPIC, 10,
        [&](JointState::ConstSharedPtr msg)
        {
            const std::vector<double> pos = msg->position;

            if (pos.size() != 2)
                return;

            EigenVector xf_new(NUM_STATES);
            xf_new << pos[0], 0.0, pos[1], 0.0;

            MPC->setDesiredState(xf_new);
            update_osqp = true;
        });

    // Set up command publisher
    using namespace std::chrono_literals;
    auto cmd_pub = node->create_publisher<JointState>(PUBLISH_TOPIC, 10);
    RCLCPP_INFO(node->get_logger(), "Publishing to JointState topic '%s'", PUBLISH_TOPIC);

    // Run MPC
    using namespace std::chrono;
    JointState::SharedPtr latest_cmd = std::make_shared<JointState>();
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

            QPSolution::Ptr qp_sol = osqp->getQPSolution();
            MPCSolution::Ptr mpc_sol = MPC->getMPCSolution(qp_sol);

            Float *u_data = mpc_sol->ustar.data();
            std::vector<Float> u_star{u_data, u_data + NUM_CONTROLS};

            std::cout << "U*: [ ";
            for (Float u : u_star)
                std::cout << u << " ";
            std::cout << "]\n";

            latest_cmd->effort = u_star;

            cmd_pub->publish(*latest_cmd);
            latest_msg = nullptr;
        });
    RCLCPP_INFO(node->get_logger(), "Running MPC with a period of %lu ms", duration_cast<milliseconds>(PUBLISH_PERIOD).count());

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}