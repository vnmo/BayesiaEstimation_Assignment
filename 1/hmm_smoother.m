function [estimStates] = hmm_smoother(numStates,observations,noiseVar,P)

N=length(observations);
states=1:numStates;
sigma=sqrt(noiseVar);
estimStates=zeros(N,1);

probDist_forward(1,:)= normpdf(observations(1),states,sigma);
[~,estimStates(1,1)]=max(probDist_forward(1,:));
% probDist(1,:)= [0,0,0];
% probDist(1,observations(1))=1;
% estimStates(1)=observations(1);

for n=2:N
    pVector = normpdf(observations(n),states,sigma).* (probDist_forward(n-1,:)*P) ;
    probDist_forward(n,:) = pVector/sum(pVector);
%     [~,estimStates(n)]=max(probDist(n,:));
end


probDist_backward=zeros(N,numStates);
probDist_backward(N,observations(N))=1;
for n=N-1:-1:1
    pVector = (probDist_backward(n+1,:).*normpdf(observations(n+1),states,sigma))*P;
    probDist_backward(n,:) = pVector/sum(pVector);
end

probDist=probDist_backward.*probDist_forward;
for n=1:N
    [~,estimStates(n,1)]=max(probDist(n,:));
end

end