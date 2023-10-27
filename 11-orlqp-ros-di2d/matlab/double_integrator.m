clc
clear
close all

runDoubleIntegrator2D();

function runDoubleIntegrator2D()

    node = ros2node("/mpc_matlab");
    pub = ros2publisher(node, '/di2d/info', 'std_msgs/Float64MultiArray');
    ros2subscriber(node, '/di2d/cmd', 'std_msgs/Float64MultiArray', @subCallback);
    goal_pub = ros2publisher(node, '/di2d/goal', 'std_msgs/Float64MultiArray');
    
    pause(1.0)

    Z0 = [1; -1; 1; -1];
    ZF = [0; 0; 0; 0];

    in_msg = ros2message('std_msgs/Float64MultiArray');
    in_msg.data = [0; 0];

    out_msg = ros2message('std_msgs/Float64MultiArray');
    
    goal_msg = ros2message('std_msgs/Float64MultiArray');
    goal_msg.data = ZF;
    send(goal_pub, goal_msg);

    figure('Color', [1 1 1], 'Position', [100 200 1200 800]);
    hold on
    grid on
    current_pos = plot(Z0(1), Z0(3), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
    desired_pos = plot(ZF(1), ZF(3), 'xr');
    xlim([-5 5])
    ylim([-5 5])

    set(gca, 'ButtonDownFcn', @clickCallback);
    set(current_pos, 'PickableParts', 'all');
    set(desired_pos, 'PickableParts', 'all');
    set(allchild(gca), 'ButtonDownFcn', @clickCallback);

    sim_time = 30;
    sim_time_step = 0.02;
    tic
    for k = 1:round(sim_time/sim_time_step)
        t = k*sim_time_step;
        u = in_msg.data;
        Z0 = Z0 + diODE(t, Z0, u, sim_time_step);
        set(current_pos, 'xdata', Z0(1), 'ydata', Z0(3));
        out_msg.data = Z0.';
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

        goal_msg.data = [x, 0, y, 0];

        set(desired_pos, 'xdata', x, 'ydata', y);
    
        send(goal_pub, goal_msg);
    end
end
