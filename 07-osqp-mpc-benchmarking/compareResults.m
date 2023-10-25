clc
clear
close all
%%

CPU_dat = readmatrix("results_CPU.csv");
GPU_dat = readmatrix("results_GPU.csv");

N = CPU_dat(:,1);
CPU_times = CPU_dat(:,2);
GPU_times = GPU_dat(:,2);

figure('Color', [1 1 1])
hold on
grid on
plot(N, CPU_times, '-b', 'LineWidth', 1.5);
plot(N, GPU_times, '-g', 'LineWidth', 1.5);
title("OSQP MPC")
xlabel("# of problems")
ylabel("Comp. Time (s)")
legend(["CPU" "GPU"], "Location","northwest")

%%

CPU_dat2 = readmatrix("results_CPU2.csv");
GPU_dat2 = readmatrix("results_GPU2.csv");

N2 = CPU_dat2(:,1);
CPU_times2 = CPU_dat2(:,2);
GPU_times2 = GPU_dat2(:,2);

figure('Color', [1 1 1])
hold on
grid on
plot(N2, CPU_times2, '-b', 'LineWidth', 1.5);
plot(N2, GPU_times2, '-g', 'LineWidth', 1.5);
title("OSQP MPC 2")
xlabel("# of problems")
ylabel("Comp. Time (s)")
legend(["CPU" "GPU"], "Location","northwest")

%%

CPU_dat3 = readmatrix("results_CPU3.csv");
GPU_dat3 = readmatrix("results_GPU3.csv");

N3 = CPU_dat3(:,1);
CPU_times3 = CPU_dat3(:,2);
GPU_times3 = GPU_dat3(:,2);

figure('Color', [1 1 1])
hold on
grid on
plot(N, CPU_times, '-b', 'LineWidth', 1.5);
plot(N, GPU_times, '-g', 'LineWidth', 1.5);
plot(N3, CPU_times3, '--b', 'LineWidth', 1.5);
plot(N3, GPU_times3, '--g', 'LineWidth', 1.5);
title("OSQP MPC 3")
xlabel("# of problems")
ylabel("Comp. Time (s)")
legend(["CPU" "GPU" "CPU w/ MT" "GPU w/ MT"], "Location","northwest")