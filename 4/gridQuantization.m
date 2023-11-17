function [mse_gridQuantization_mean] = gridQuantization(a,var_w,var_v,observations,grid_resolution, num_steps)

observations = observations';

grid = -5:grid_resolution:5; % Create a grid of possible state values
grid_estimates_mean = zeros(1,num_steps);
grid_estimates_map = zeros(1,num_steps);

for k = 2:num_steps
    % Realising x
    grid_states = a*grid + sqrt(var_w)*randn(1,length(grid));

    % Likelihood of y: Update
    likelihood = normpdf(observations(k),atan(grid_states),sqrt(var_v));

    % Assuming a uniform prior
    posterior = likelihood .* normpdf(grid_states,grid,sqrt(var_w));

    % Normalize the posterior
    posterior = posterior / sum(posterior);

    % Estimate the state (Mean of the posterior)
    grid_estimates_mean(k) = grid * posterior';

    % Estimate the state (Mode of the posterior)
    %     [~, idxMaxPosterior] = max(posterior);
    %     grid_estimates_map(k) =  grid(idxMaxPosterior);
end

% Calculate Mean Square Errors
mse_gridQuantization_mean = mean((observations - grid_estimates_mean).^2);
% mse_gridQuantization_map = mean((observations - grid_estimates_map).^2);

end