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
plot(N, CPU_times, '-b');
plot(N, GPU_times, '-g');
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
plot(N, CPU_times, '-b');
plot(N, GPU_times, '-g');
plot(N2, CPU_times2, '-b');
plot(N2, GPU_times2, '-g');
title("OSQP MPC 2")
xlabel("# of problems")
ylabel("Comp. Time (s)")