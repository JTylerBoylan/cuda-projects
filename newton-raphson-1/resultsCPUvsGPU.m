%% CPU vs GPU Newton-Raphson Benchmark
% jonathan boylan

clc
clear
close all
%%

results = readmatrix("CPUvsGPU.csv");

N = results(:, 1);
CPU_ms = results(:, 2) * 1E-6;
GPU_ms = results(:, 3) * 1E-6;

CPU_line = polyfit(N, CPU_ms, 1);
GPU_line = polyfit(N, GPU_ms, 1);

CPU_rate = 1/CPU_line(1)/1E3; % Kilo-solves/ms
GPU_rate = 1/GPU_line(1)/1E3; % Kilo-solves/ms

fprintf("CPU computation: %.2fK solves/ms\n", CPU_rate);
fprintf("GPU computation: %.1fK solves/ms\n", GPU_rate);
fprintf("GPU:CPU speedup: x%.1f\n", GPU_rate/CPU_rate);

figure('Color', [1 1 1])
hold on
grid on
plot(N,CPU_ms,'b')
plot(N,GPU_ms,'g')
xlabel("# of parabolas")
ylabel("Comp. Time (ms)")
title("CPU vs GPU")

leg_CPU = sprintf("CPU: %.2fK solves/ms", CPU_rate);
leg_GPU = sprintf("GPU: %.1fK solves/ms", GPU_rate);
legend([leg_CPU, leg_GPU])