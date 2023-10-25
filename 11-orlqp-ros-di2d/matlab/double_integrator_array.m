clc
clear
close all

num_problems = 20;

runDoubleIntegrator2DArray(num_problems);

function runDoubleIntegrator2DArray(num_problems)

    node = ros2node("/mpc_array_matlab");
    pub = ros2publisher(node, '/di2d_array/info', 'std_msgs/Float64MultiArray');
    ros2subscriber(node, '/di2d_array/cmd', 'std_msgs/Float64MultiArray', @subCallback);
    goal_pub = ros2publisher(node, '/di2d_array/goal', 'std_msgs/Float64MultiArray');

    pause(1.0)

    Z0 = (rand(4, num_problems) - 0.5) * 10.0;
    ZF = zeros(4, num_problems);

    in_msg = ros2message('std_msgs/Float64MultiArray');
    in_msg.data = zeros(1, num_problems * 2);

    out_msg = ros2message('std_msgs/Float64MultiArray');

    goal_msg = ros2message('std_msgs/Float64MultiArray');
    ZFf = reshape(ZF, [1 4*num_problems]);
    goal_msg.data = ZFf;
    send(goal_pub, goal_msg);

    figure
    hold on
    grid on
    current_pos = plot(Z0(1, :), Z0(3, :), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
    desired_pos = plot(ZF(1, :), ZF(3, :), 'xr');
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

        u = reshape(in_msg.data, [2 num_problems]);

        Z0 = Z0 + diODE(t, Z0, u, sim_time_step);

        set(current_pos, 'xdata', Z0(1, :), 'ydata', Z0(3, :));

        out_msg.data = reshape(Z0, [1 4*num_problems]);

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
    
        goal_i = [x; 0; y; 0];
        goal_r = repmat(goal_i, [1 num_problems]);
        goal_f = reshape(goal_r, [1 4*num_problems]);

        goal_msg.data = goal_f;

        set(desired_pos, 'xdata', goal_r(1,:) , 'ydata', goal_r(3,:));
    
        send(goal_pub, goal_msg);
    end
end
