function [Q, pol, nbReplayCycle, duration, memorySweeps, memorySide, sequence, bufferRPE] = PI(M, R, sequence, sequenceTraj, replayMethod)
% This function applies Policy Iteration to compute the
% optimal model-based Q-function based on the agent's estimated
% reward function hatR and transition function hatP (in the model).
% It returns the optimal Q-function and the corresponding (optimal) policy
% for the agent R based on the agent's gamma value.
%
% INPUTS:
%     M is a structure containing the MDP
%     sequence is the buffer of 0, 1 or N ordered states to be replayed
%     R is a structure containing the learning agent
%     sequence is the buffer of N states to be replayed (with an order that
%     has been predetermined based on the replayMethod)
%     replayMethod is the chosen method for replays
%
% OUTPUTS:
%     Q is the infered Q-function
%     pol is the deduced policy
%     nbReplayCycle is the number of cycles of replay performed
%     duration is the total duration of the replay (in timesteps)
%     memorySweeps stores for each replay cycle the side of the maze where
%     the sweep of states finished when reaching back the central arm
%     memorySide stores for each replay cycle the proportion of states that
%     were replayed either on the left or on the right arm
%     sequence is the updated buffer of N ordered states that have been replayed
%     bufferRPE is the updated buffer of reward prediction errors ordered
%     according to their absolute value (for prioritized sweeping method)
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 7 May 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

%% Initialization
% Copy the current policy from the agent's structure
pol = R.pol;

% Copy the table containing Q-values from the agent's structure
Q = R.Q; % zeros(R.nS,R.nA);

% sequence can contain a sampled trajectory
switch (size(sequence,2))
    case 0
        sequence = zeros(7,R.window); % 54 states
        bufferRPE = [];
    case 1 % trajectory sampling (sequence contains only starting point)
        bufferRPE = sequence;
        if (replayMethod == 18)
            sequence = [];
        end
    otherwise % sequence already contains the full ordered buffer to replay
        bufferRPE = sequence;
        sequence = [];
end

% Initialize the buffer where we store sweep characteristics
memorySweeps = []; % we store L or R for each sweep
memorySide = []; % we store L or R for each state replayed

%% Policy iteration loop
quit = 0;
nbReplayCycle = 0;
duration = 0;
while (~quit)
    nbReplayCycle = nbReplayCycle + 1;
    duration = duration + size(sequence,2);

    % storing the current policy pol
    pol2 = pol;
    
    % Evaluting the current policy
    carreP = zeros(R.nS, R.nS);
    vectR = zeros(R.nS, 1);
    for xxx=1:R.nS
        vectR(xxx) = R.hatR(xxx, pol(xxx));
        for yyy=1:R.nS
            carreP(xxx, yyy) = R.hatP(xxx, pol(xxx), yyy);
        end;
    end;
    %V = inv(eye(R.nS, R.nS) - R.gamma * carreP) * vectR;
    V = (eye(R.nS, R.nS) - R.gamma * carreP) \ vectR;
    
    switch (replayMethod)
        case 18 %% MB prioritized sweeping combined with PI
            if (~isempty(bufferRPE))
                % we store so as to remember to visit all "surprising" states during the next trajectory sampling with PI
                bufferCopy = bufferRPE;
                [Q, sequence, bufferRPE] = prioritizedSweeping(R, sequence, bufferRPE, replayMethod, V, Q);
            else
                bufferCopy = [];
            end
            
            %% Performing PI update with trajectory sampling
            % strategy 0: loop on all states and actions like classical PI
%             for xxx=1:R.nS
%                 for uuu=1:R.nA
%                     Q(xxx, uuu) = R.hatR(xxx, uuu) + R.gamma * sum(reshape(R.hatP(xxx, uuu, :), R.nS, 1)' * V);
%                 end
%             end
            % strategy 1, 2 or 3: trajectory sampling
            sequence = [sequence sequenceTraj];
            [Q, sequence, memorySweeps] = trajectorySampling(R, sequence, bufferCopy, memorySweeps, replayMethod, V, Q);
             
        otherwise % incl. case 2 = basic PI
            % Performing PI update
            for xxx=1:R.nS
                for uuu=1:R.nA
                    Q(xxx, uuu) = R.hatR(xxx, uuu) + R.gamma * sum(reshape(R.hatP(xxx, uuu, :), R.nS, 1)' * V);
                end
            end
    end

    % Deducing the policy
    pol = zeros(R.nS, 1);
    for x=1:R.nS
        pol(x) = argmax(Q(x, :));
    end
    
    % Stopping condition when the policy has converged
    variation = sum(abs(pol2-pol));
    if (variation == 1) % if variation is due to negligible difference between 2 Q-values
        negligibleDifference = (Q(argmax(abs(pol-pol2)>0),pol(argmax(abs(pol-pol2)>0)))-Q(argmax(abs(pol-pol2)>0),pol2(argmax(abs(pol-pol2)>0)))) < 10e-4;
    end
    if ((variation < R.replayiterthreshold)||((variation==1)&&(negligibleDifference))||((R.replaybudget > 0)&&(nbReplayCycle >= R.replaybudget))) % < 0.1
        quit = 1;
%     else
%         if (~isempty(bufferCopy))
%             bufferCopy(end,:) = 0; % we reset all surprising states to "not visited" for trajectory sampling
%         end
    end
end

