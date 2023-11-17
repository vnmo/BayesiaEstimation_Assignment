clc;clear;close all;

% Parameters
a = 0.6;
num_particles = 50000;
num_steps = 100;
var_w = 1;
var_v = 1;
grid_resolution = 0.001;

% Simulated Observationsx
x = zeros(num_steps,1);
y = zeros(num_steps,1);
x(1) = rand;
y(1) = atan(x(1))+sqrt(var_v)*randn;
for k = 2:num_steps
    x(k) = a*x(k-1)+sqrt(var_w)*randn;
    y(k) = atan(x(1))+sqrt(var_v)*randn;
end
observations = y;

mse_particleFilter = particleFilter(a, var_w, var_v, observations, num_steps, num_particles);
mse_gridQuantization_mean = gridQuantization(a, var_w, var_v, observations, grid_resolution, num_steps);

