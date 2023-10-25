clc
clear
close all

runROSNode();

function runROSNode()

    node = ros2node("/mpc_matlab");
    pub = ros2publisher(node, '/lip/info', 'sensor_msgs/JointState');
    ros2subscriber(node, '/lip/cmd', 'sensor_msgs/JointState', @subCallback);
    goal_pub = ros2publisher(node, '/lip/goal', 'sensor_msgs/JointState');
    
    in_msg = ros2message("sensor_msgs/JointState");
    out_msg = ros2message("sensor_msgs/JointState");
    in_msg.effort = [0; 0];

    Z0 = [1; -1; 1; -1];
    ZF = [0; 0; 0; 0];

    figure
    hold on
    grid on
    current_pos = plot(Z0(1), Z0(3), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
    desired_pos = plot(ZF(1), ZF(3), 'xr');
    xlim([-5 5])
    ylim([-5 5]);

    set(gca, 'ButtonDownFcn', @clickCallback);
    set(current_pos, 'PickableParts', 'all');
    set(desired_pos, 'PickableParts', 'all');
    set(allchild(gca), 'ButtonDownFcn', @clickCallback);

    sim_time = 30;
    sim_time_step = 0.02;
    tic
    for k = 1:round(sim_time/sim_time_step)
        t = k*sim_time_step;
        u = in_msg.effort;
        Z0 = Z0 + diODE(t, Z0, u, sim_time_step);
        set(current_pos, 'xdata', Z0(1), 'ydata', Z0(3));
        out_msg.position = [Z0(1) Z0(3)];
        out_msg.velocity = [Z0(2) Z0(4)];
        send(pub, out_msg);
        pause(t - toc);
    end
    
    function dZ = diODE(~, Z, U, T)
        A = [0 T 0 0;
             0 0 0 0;
             0 0 0 T;
             0 0 0 0];
        B = [0 0;
             T 0;
             0 0;
             0 T];
        dZ = A*Z + B*U;
    end

    function subCallback(msg)
        in_msg = msg;
    end

    function clickCallback(~, ~)
        point = get(gca, 'CurrentPoint');
        x = point(1, 1);
        y = point(1, 2);
    
        goal_msg = ros2message("sensor_msgs/JointState");
        goal_msg.position = [x, y];

        set(desired_pos, 'xdata', x, 'ydata', y);
    
        send(goal_pub, goal_msg);
    end
end
