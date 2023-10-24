clc
clear
close all
%%

Z0 = [1; -1; 1; -1];
ZF = [0; 0; 0; 0];

horizon = 1.0;

figure
hold on
grid on
current_pos = plot(Z0(1), Z0(3), 'og');
desired_pos = plot(ZF(1), ZF(3), 'xr');
xlim([-5 5])
ylim([-5 5]);

sim_time = 10;
sim_time_step = 0.02;
tic
for k = 1:round(sim_time/sim_time_step)

    t = k*sim_time_step;
    Z0 = Z0 + diODE(t, Z0, [0; 0], sim_time_step);

    set(current_pos, 'xdata', Z0(1), 'ydata', Z0(3));

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