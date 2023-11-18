% Define parameters
N = 1000; % Number of points
numStates = 3; % Number of states
noiseVariances = [0.1, 0.5, 1, 2]; % Different noise variances to test

% Initialize variables to store results
mseFilter = zeros(length(noiseVariances), 1);
mseSmoother = zeros(length(noiseVariances), 1);

for v = 1:length(noiseVariances)
    noiseVariance = noiseVariances(v);
    
    % Generate random transition matrix for the Markov chain
    transitionMatrix = rand(numStates, numStates);
    transitionMatrix = transitionMatrix ./ sum(transitionMatrix, 2);

    % Generate observations with Gaussian noise
    trueStates = zeros(N, 1);
    observations = zeros(N, 1);
    trueStates(1) = randi(numStates);
    observations(1) = randn * sqrt(noiseVariance);
    for t = 2:N
        trueStates(t) = randsample(1:numStates, 1, true(transitionMatrix(trueStates(t-1), :)));
        observations(t) = trueStates(t) + randn * sqrt(noiseVariance);
    end

    % HMM filtering
    estimatedStatesFilter = hmmestimate(observations, transitionMatrix, ones(numStates, 1), ones(numStates, 1));

    % HMM smoothing
    estimatedStatesSmoother = hmmsmooth(observations, transitionMatrix, ones(numStates, 1), ones(numStates, 1));

    % Calculate Mean Square Error for filter and smoother
    mseFilter(v) = mean((estimatedStatesFilter - trueStates').^2);
    mseSmoother(v) = mean((estimatedStatesSmoother - trueStates').^2);
end

% Plot results
figure;
plot(noiseVariances, mseFilter, 'r-o', 'LineWidth', 2);
hold on;
plot(noiseVariances, mseSmoother, 'b-o', 'LineWidth', 2);
xlabel('Noise Variance');
ylabel('Mean Squared Error');
legend('Filter', 'Smoother');
title('Mean Squared Error of Filter and Smoother');
grid on;
