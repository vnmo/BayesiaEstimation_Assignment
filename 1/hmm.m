clc
clear

% https://www.researchgate.net/publication/331917314_Chapter_11_Hidden_Markov_Model_Filtering_and_Smoothing
% Define parameters
N = 1000; % Number of points
numStates = 3; % Number of states
NOISEVAR = 0.1:0.1:5; % Different noise variances to test

% Initialize variables to store results
% mseFilter = zeros(length(NOISEVAR), 1);
% mseSmoother = zeros(length(NOISEVAR), 1);

% Generate random transition matrix for the Markov chain
%  transitionP = rand(numStates, numStates);
transitionP = [0.3022  ,  0.4888  ,  0.1100;0.2565  ,  0.2475  ,  0.3090;    0.1619,    0.0547,    0.4172];

transitionP = transitionP ./ sum(transitionP, 2);
for repeat=1:20
    repeat

    trueStates = zeros(N, 1);
    trueStates(1) = randi(numStates);
    for t = 2:N
        rnum=rand;
        transitionCDF=cumsum(transitionP(trueStates(t-1), :));
        for j=1:numStates
            if(rnum<=transitionCDF(j))
                trueStates(t)=j;
                break;
            end
        end
    end

    for v = 1:length(NOISEVAR)
        noiseVar = NOISEVAR(v);

        observations = trueStates+ randn(N,1)*sqrt(noiseVar);
        observations(1)=trueStates(1);
        observations(N)=trueStates(N);

        estimStates_filter=hmm_filter(numStates,observations,noiseVar,transitionP);
        estimStates_smoother=hmm_smoother(numStates,observations,noiseVar,transitionP);


        mseFilter(repeat,v) = mean((estimStates_filter - trueStates).^2);
        mseSmoother(repeat,v) = mean((estimStates_smoother - trueStates).^2);
    end
end

mseFilter=mean(mseFilter);
mseSmoother=mean(mseSmoother);

% Plot results

close all;
plot(NOISEVAR, mseFilter, '-o', 'LineWidth', 2);
hold on;
plot(NOISEVAR, mseSmoother, '-*', 'LineWidth', 2);
xlabel('Noise Variance');
ylabel('Mean Squared Error');
legend('HMM Filter', 'HMM Smoother');
grid on;


% figure;
% stairs(trueStates(1:25),'--', 'LineWidth', 2)
% hold;
% stairs(estimStates_filter(1:25),'-o', 'LineWidth', 2)
% stairs(estimStates_smoother(1:25),'-*', 'LineWidth', 2)
% legend('Ground truth', 'HMM Filter', 'HMM smoother');


