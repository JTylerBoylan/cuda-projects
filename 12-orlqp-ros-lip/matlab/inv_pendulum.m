clc
clear
close all

g = 9.81;
M = 0.5;
m = 0.5;
b = 0.1;
l = 0.3;
I = 0.006;
runInvertedPendulum(g, M, m, b, l, I);

function runInvertedPendulum(g, M, m, b, l, I)

    D = I*(M + m) + M*m*l*l;
    A22 = -(I+m*l*l)*b / D;
    A23 = m*m*g*l*l / D;
    A42 = -m*l*b / D;
    A43 = m*g*l*(M + m) / D;
    B21 = (I + m*l*l) / D;
    B41 = m*l / D;

    node = ros2node("/mpc_matlab");
    pub = ros2publisher(node, '/lip/info', 'std_msgs/Float64MultiArray');
    ros2subscriber(node, '/lip/cmd', 'std_msgs/Float64MultiArray', @subCallback);
    goal_pub = ros2publisher(node, '/lip/goal', 'std_msgs/Float64MultiArray');
    
    pause(1.0)

    Z0 = [0; 0; 0; 0];
    ZF = [1; 0; 0; 0];

    in_msg = ros2message('std_msgs/Float64MultiArray');
    in_msg.data = 0;

    out_msg = ros2message('std_msgs/Float64MultiArray');
    
    goal_msg = ros2message('std_msgs/Float64MultiArray');
    goal_msg.data = ZF;
    send(goal_pub, goal_msg);

    cart_width = 0.5;
    cart_height = 0.5;
    cart_body = [+cart_width/2, -cart_width/2, -cart_width/2, +cart_width/2;
                 +cart_height/2, +cart_height/2, -cart_height/2, -cart_height/2];
    pend_width = 0.1;
    pend_height = 0.6;
    pend_body = [+pend_width/2, -pend_width/2, -pend_width/2, +pend_width/2;
                 +pend_height, +pend_height, 0, 0];
    R_pend = @(phi) [cos(phi), -sin(phi); sin(phi), cos(phi)];

    figure
    hold on
    grid on

    cart_plot = fill(0, 0, 'r');
    pend_plot = fill(0, 0, 'g');
    drawSystem();
    goal_plot = plot([0 0], [-5 5], '--k');
    drawGoal();

    xlim([-3 3])
    ylim([-1 3]);

    set(gca, 'ButtonDownFcn', @clickCallback);
    set(cart_plot, 'PickableParts', 'all');
    set(pend_plot, 'PickableParts', 'all');
    set(goal_plot, 'PickableParts', 'all');
    set(allchild(gca), 'ButtonDownFcn', @clickCallback);

    sim_time = 300;
    sim_time_step = 0.02;
    tic
    for k = 1:round(sim_time/sim_time_step)
        t = k*sim_time_step;
        u = in_msg.data;
        Z0 = Z0 + diODE(t, Z0, u, sim_time_step);
        out_msg.data = Z0.';
        send(pub, out_msg);
        drawSystem();
        pause(t - toc);
    end

    function drawSystem()
        cart_pos = [Z0(1); 0];
        cart_body_w = cart_body + cart_pos;
        set(cart_plot, 'xdata', cart_body_w(1,:), 'ydata', cart_body_w(2,:));
        pend_pos = [0; +cart_height/2];
        pend_R = R_pend(Z0(3));
        pend_body_w = pend_R*pend_body + pend_pos + cart_pos;
        set(pend_plot, 'xdata', pend_body_w(1,:), 'ydata', pend_body_w(2,:));
    end

    function drawGoal()
        x = ZF(1);
        set(goal_plot, 'xdata', [x x]);
    end
    
    function dZ = diODE(~, Z, U, T)
        A = [0 T 0 0;
             0 T*A22 T*A23 0;
             0 0 0 T;
             0 T*A42 T*A43 0];
        B = [0;
             T*B21;
             0;
             T*B41];
        dZ = A*Z + B*U;
    end

    function subCallback(msg)
        in_msg = msg;
    end

    function clickCallback(~, ~)
        point = get(gca, 'CurrentPoint');
        x = point(1, 1);
        ZF = [x; 0; 0; 0];
        goal_msg.data = ZF.';
        send(goal_pub, goal_msg);
        drawGoal();
    end
end
