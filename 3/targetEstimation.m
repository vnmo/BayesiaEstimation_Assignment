clc;clear;close all

% Config
t=1;
num_steps = 100;
acc_x = 0.2; % known acceleration in x dir
acc_y = 0.2; % known acceleration in y dir

A = [1, t, 0, 0; ...
     0, 1, 0, 0; ...
     0, 0, 1, t; ...
     0, 0, 0, 1];

f = [t^2/2, 0     ; ...
     t    , 0     ; ...
     0    , t^2/2 ; ...
     0    , t     ];

C = [1, 0, 0, 0; ...
     0, 0, 1, 0]; % Observation matrix

Q = eye(4); % Covariance of te noise w_k on state change. 
            % Let it be uncorrelated.
mu_q = zeros(4, 1);
R = eye(2); % noise covariance of v_k on position observation.
            % Let it be uncorrelated.
mu_r = zeros(2, 1);

r = [acc_x; acc_y];


state_z = zeros(4, num_steps);
state_y = zeros(2, num_steps);
state_z(:, 1) = [0; 0; 0; 0]; % Let the ship start at origin


% Simulate ground truth and observed ship position
for k = 1:num_steps-1    
    state_z(:,k+1) = A*state_z(:,k) + f*r + transpose(mvnrnd(mu_q, Q));
    state_y(:,k+1) = C*state_z(:,k+1) + transpose(mvnrnd(mu_r, R));
end

% Kalman filter
x_est = zeros(4, num_steps);
covarMat = eye(4); % Initial state covariance

% Kalman filter loop 
% Ref: https://en.wikipedia.org/wiki/Kalman_filter
for k = 2:num_steps
    % Prediction 
    x_est_cap = A * x_est(:, k-1)+ f*r; %state 
    covarMat_cap = A * covarMat * A' + Q;% error covariance matrix
    
    % Update
    K = covarMat_cap * C' * inv(C * covarMat_cap * C' + R);
    x_est(:, k) = x_est_cap + K * (state_y(:, k) - C * x_est_cap);
    covarMat = (eye(4) - K * C) * covarMat_cap;
end


% Plotting
figure;
plot(1:num_steps, state_z(1, :), 'b-','linewidth',3);
hold on;
plot(1:num_steps, x_est(1, :), 'r--','linewidth',3);
plot(1:num_steps, state_z(3, :), 'g-','linewidth',3);
plot(1:num_steps, x_est(3, :), 'm--','linewidth',3);

legend('Actual position dim-1', 'Estimated position dim-1','Actual position dim-2', 'Estimated position dim-2','fontsize',30,'interpreter','latex','location','northwest');
xlabel('$t/T_s$','fontsize',30,'interpreter','latex');
ylabel('Position','fontsize',30,'interpreter','latex');
grid on


figure;
plot(1:num_steps, state_z(2, :), 'b-','linewidth',3);
hold on;
plot(1:num_steps, x_est(2, :), 'r--','linewidth',3);
plot(1:num_steps, state_z(4, :), 'g-','linewidth',3);
plot(1:num_steps, x_est(4, :), 'm--','linewidth',3);

legend('Actual velocity dim-1', 'Estimated velocity dim-1','Actual velocity dim-2', 'Estimated velocity dim-2','fontsize',30,'interpreter','latex','location','northwest');
xlabel('$t/T_s$','fontsize',30,'interpreter','latex');
ylabel('Velocity','fontsize',30,'interpreter','latex');
grid on