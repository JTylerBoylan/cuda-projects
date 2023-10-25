clc
clear
close all
%%

tolerance = -4:1:0;

frequency = [9.5, 12, 16, 20, 25];
squared_error = [0.00017, 0.00138, 0.0126, 0.0406, 0.142];

figure
plot(tolerance, frequency)
xlabel("Tolerance (10^x)")
ylabel("Frequency (Hz)")

figure
plot(tolerance, squared_error)
xlabel("Tolerance (10^x)")
ylabel("Squared error")