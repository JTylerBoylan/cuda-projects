%% CPU vs GPU Newton-Raphson Benchmark
% jonathan boylan

clc
clear
close all
%%

results = readmatrix("compare.csv");

N = results(:, 1);
CPU_ms = results(:, 2) * 1E-6;
GPU_ms = results(:, 3) * 1E-6;

figure('Color', [1 1 1])
hold on
grid on
plot(N,CPU_ms,'b')
plot(N,GPU_ms,'g')
xlabel("# of parabolas")
ylabel("Comp. Time (ms)")
legend(["CPU" "GPU"])

% remove outliers
GPU_ms(GPU_ms > 2) = nan;

figure('Color', [1 1 1])
hold on
grid on
plot(N, GPU_ms, 'g')
xlabel("# of parabolas")
ylabel("Comp. Time (ms)")
legend("GPU")