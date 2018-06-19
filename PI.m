function [Q, pol, nbReplayCycle, duration] = PI(R, sequence)
% This function applies Policy Iteration to compute the
% optimal model-based Q-function based on the agent's estimated
% reward function hatR and transition function hatP (in the model).
% It returns the optimal Q-function and the corresponding (optimal) policy
% for the agent R based on the agent's gamma value.
%
% INPUTS:
%     R is a structure containing the learning agent
%     sequence is the buffer of N states to be replayed (with an order that
%     has been predetermined based on the replayMethod)
%
% OUTPUTS:
%     Q is the infered Q-function
%     pol is the deduced policy
%     nbReplayCycle is the number of cycles of replay performed
%     duration is the total duration of the replay (in timesteps)
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

%% Initialization
% Copy the current policy from the agent's structure
pol = R.pol;

% Initialize a Q-function which does not depend on the agent's Q
Q = zeros(R.nS,R.nA);

% sequence can contain a sampled trajectory
if (isempty(sequence))
    sequence = zeros(1,R.window); % 54 states
end

% Policy iteration loop
quit = 0;
nbReplayCycle = 0;
duration = 0;
while (~quit)
    nbReplayCycle = nbReplayCycle + 1;
    duration = duration + size(sequence,2);

    % storing the current policy pol
    pol2 = pol;
    
    % Evaluting the current policy
    carreP = zeros(R.nS,R.nS);
    vectR = zeros(R.nS,1);
    for x=1:R.nS
        vectR(x) = R.hatR(x,pol(x));
        for y=1:R.nS
            carreP(x,y) = R.hatP(x,pol(x),y);
        end;
    end;
    V = inv(eye(R.nS,R.nS) - R.gamma * carreP) * vectR;

    % Performing PI update
    for x=1:R.nS
        for u=1:R.nA
            Q(x,u) = R.hatR(x,u) + R.gamma * sum(reshape(R.hatP(x,u,:),R.nS,1)' * V);
        end;
    end;

    % Deducing the policy
    pol = zeros(R.nS,1);
    for x=1:R.nS
        pol(x) = argmax(Q(x,:));
    end;
    
    % Stopping condition when the policy has converged
    variation = sum(abs(pol2-pol));
    if ((variation < R.replayiterthreshold)||((R.replaybudget > 0)&&(nbReplayCycle >= R.replaybudget))) % < 0.1
        quit = 1;
    end
end

