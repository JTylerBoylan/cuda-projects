%% GPU only Newton-Raphson Benchmark
% jonathan boylan

clc
clear
close all
%%

results = readmatrix("GPUonly.csv");

N = results(:, 1);
GPU_ms = results(:, 2) * 1E-6;

GPU_line = polyfit(N, GPU_ms, 1);

GPU_rate = 1/GPU_line(1)/1E3; % Kilo-solves/ms
fprintf("GPU computation: %.2fK solves/ms\n", GPU_rate);

figure('Color', [1 1 1])
hold on
grid on
plot(N,GPU_ms,'g')
plot(N,N.*GPU_line(1) + GPU_line(2), '--g')
xlabel("# of parabolas")
ylabel("Comp. Time (ms)")

leg_GPU = sprintf("GPU Experimental Data");
leg_GPU_poly = sprintf("Best fit: %.1fK solves/ms", GPU_rate);
legend([leg_GPU leg_GPU_poly])