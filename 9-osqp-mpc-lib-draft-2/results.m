clc
clear
close all
%%

data = readmatrix("results.csv");

N = data(:,1);
kHz = data(:,2);
T_ms = 1./kHz;

figure('Color', [1 1 1])
yyaxis right
plot(N,T_ms, 'LineWidth', 2)
ylabel("Time (ms)", 'FontWeight','bold')
yyaxis left
plot(N,kHz, 'LineWidth', 2)
ylabel("Frequency (kHz)",'FontWeight','bold')
xlabel("Number of nodes",'FontWeight','bold')
