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

fprintf("CPU computation: %d solves/ms\n", round(1/CPU_line(1)));
fprintf("GPU computation: %d solves/ms\n", round(1/GPU_line(1)));
fprintf("GPU:CPU speedup: x%.1f\n", CPU_line(1)/GPU_line(1));

figure('Color', [1 1 1])
hold on
grid on
plot(N,CPU_ms,'b')
plot(N,GPU_ms,'g')
xlabel("# of parabolas")
ylabel("Comp. Time (ms)")
legend(["CPU: 28.5K solves/ms" "GPU: 320K solves/ms"])