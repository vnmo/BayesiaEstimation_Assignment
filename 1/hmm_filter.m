function [estimStates] = hmm_filter(numStates,observations,noiseVar,P)

N=length(observations);
states=1:numStates;
sigma=sqrt(noiseVar);
estimStates=zeros(N,1);

probDist(1,:)= normpdf(observations(1),states,sigma);
[~,estimStates(1,1)]=max(probDist(1,:));
% probDist(1,:)= [0,0,0];
% probDist(1,observations(1))=1;
% estimStates(1)=observations(1);

for n=2:N
    probDist(n,:) = normpdf(observations(n),states,sigma).* (probDist(n-1,:)*P) ;
    [~,estimStates(n,1)]=max(probDist(n,:));
end

end