function [Q, pol, nbReplayCycle, duration, memorySweeps, memorySide, sequence, bufferRPE] = VI(M, R, bufferRPE, sequenceTraj, replayMethod)
% This function applies Value Iteration to infer the estimated
% optimal model-based Q-function based on the agent's estimated
% reward function hatR and transition function hatP (in the model).
% It returns the optimal Q-function and the corresponding (optimal) policy
% for the agent R based on the agent's gamma value.
%
% INPUTS:
%     M is a structure containing the MDP
%     sequence is the buffer of 0, 1 or N ordered states to be replayed
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
%     last modified 20 May 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

%% Initialization
% Copy the table containing Q-values from the agent's structure
Q = R.Q;

% sequence will contain the sampled replay trajectory
sequence = [];

% Initialize the buffer where we store sweep characteristics
memorySweeps = []; % we store L or R for each sweep
memorySide = []; % we store L or R for each state replayed

% Value iteration loop
quit = 0;
nbReplayCycle = 0;
duration = 0;
while (~quit)
    nbReplayCycle = nbReplayCycle + 1;
    duration = duration + R.window;

    % Storing the current Q function
    Q2 = Q;
    
    % Updating the Q function
    switch (replayMethod)
        case {6,11,12,17,19,20} %% MB/DYNA prioritized sweeping
            if (~isempty(bufferRPE))
                bufferCopy = bufferRPE;
                [Q, sequence, bufferRPE] = prioritizedSweeping(R, sequence, bufferRPE, replayMethod, 0, Q);
            else
                bufferCopy = [];
            end
            
            %% Performing VI update with trajectory sampling
            if (replayMethod == 19)
                % strategy 0: loop on all states and actions like classical PI
    %             for xxx=1:R.nS
    %                 for uuu=1:R.nA
    %                     Q(xxx, uuu) = R.hatR(xxx, uuu) + R.gamma * sum(reshape(R.hatP(xxx, uuu, :), R.nS, 1)' * V);
    %                 end
    %             end
                % strategy 1, 2 or 3: trajectory sampling
                sequence = [sequence sequenceTraj];
                [Q, sequence, memorySweeps] = trajectorySampling(R, sequence, bufferRPE, memorySweeps, replayMethod, 0, Q);
            end
            
        case {8,14} %% MB-RL/Dyna-RL shuffled: VI with shuffled state
            newsweep = false; % each time we go through decision point,
            % we consider that a new sweep starts.
            % each time we enter the central arm, newsweep is reset
            for iii=1:R.window
                x = M.stata(randi(length(M.stata))); % randi(R.nS);
                % Ask the agent which action to perform
                acta = possibleMoves(x, 0);
                % Softmax action selection
                %u = valueBasedDecision(Q(x,acta), R.decisionRule, R.beta, 0);
                %u = acta(u);
                % Random action selection
                u = acta(randi(length(acta)));
                sequence = [sequence [x ; u ; zeros(5,1)]];
                sequence(3,end) = R.hatR(x,u);
                sequence(6,end) = argmax(R.hatP(x,u,:));
                % Update Q
                Qmax = max(Q,[],2);
                if (replayMethod == 14) % DYNA thus MF update
                    y = drand01(reshape(R.hatP(x,u,:),1,R.nS));
                    RPE = R.hatR(x,u) + R.gamma * Qmax(y) - Q(x, u);
                    Q(x, u) = Q(x, u) + R.alpha * RPE;
                    sequence(6,end) = y;
                else % MB thus MB update 
                    Q(x, u) = R.hatR(x, u) + R.gamma * sum(reshape(R.hatP(x, u, :), R.nS, 1) .* Qmax);
                end
                % if it's a new sweep and we reach the central arm, we store the
                % characteristics of this sweep (L/R).
                if (newsweep&&(sequence(6,end)==24)) % old : state 24 then 26
                    newsweep = false;
                    % we store the side of the sweep
                    jjj = size(sequence,2);
                    % we loop until the first state found outside the central arm
                    while ((jjj > 1)&&((sequence(1,jjj)==15)||(sequence(1,jjj)==21)||(sequence(1,jjj)==22)||(sequence(1,jjj)==23)||(sequence(1,jjj)==24)||(sequence(1,jjj)==25)||(sequence(1,jjj)==26)||(sequence(1,jjj)==27))) %(sequence(1,jjj)==sequence(7,iii)))
                        jjj = jjj - 1;
                    end
                    sweepside = 24;
                    if ((sequence(1,jjj)<14)||(sequence(1,jjj)==18)||(sequence(1,jjj)==19))
                        % we are in the left arm
                        sweepside = 18;
                    end
                    if (sequence(1,jjj)>29)
                        % we are in the right arm
                        sweepside = 30;
                    end
                    %[jjj sequence(1,jjj)]
                    memorySweeps = [memorySweeps sequence(1,jjj)]; % 18, 30 or 24?
                end
                % if we arrive at decision-point, we initiate a new sweep
                if (~newsweep&&(sequence(6,end)==26))
                    newsweep = true;
                    %iii
                end
            end
            
        case {10,15,18} % MB/DYNA trajectory sampling
            sequence = [sequence sequenceTraj];
            [Q, sequence, memorySweeps] = trajectorySampling(R, sequence, [], memorySweeps, replayMethod, 0, Q);
            
        case {13,16} % MB/DYNA bidirectional planning
            sequence = [sequence sequenceTraj];
            newsweep = false; % each time we go through decision point,
            backwardDirection = true; % backward until starting position, then forward
            % we consider that a new sweep starts.
            % each time we enter the central arm, newsweep is reset
            iii = 1;
            while (iii <= R.window)
                if (backwardDirection)
                    x = sequence(1,end);
                    predFound = false;
                    for aaa = 1:R.nA
                        pred = R.hatP(:,aaa,x)>(1/R.nS);
                        indexes = (1:R.nS);
                        pred = indexes(pred); %pred .* (1:R.nS);
                        %pred = pred(pred>0); % predecessors have led at least one time to x with aaa
                        while (~isempty(pred))
                            sequence = [sequence [pred(1) ; aaa ; 0 ; 0 ; 0 ; x ; 0]];
                            iii = iii + 1;
                            % Update Q
                            Qmax = max(Q,[],2);
                            if (replayMethod == 16) % DYNA thus MF update
                                RPE = R.hatR(pred(1),aaa) + R.gamma * Qmax(x) - Q(pred(1), aaa);
                                Q(pred(1), aaa) = Q(pred(1), aaa) + R.alpha * RPE;
                            else % MB thus MB update 
                                Q(pred(1), aaa) = R.hatR(pred(1), aaa) + R.gamma * sum(reshape(R.hatP(pred(1), aaa, :), R.nS, 1) .* Qmax);
                            end
                            pred(1) = [];
                        end
                    end
                    if ((x == sequence(6,1))||(~predFound)) % starting position
                        backwardDirection = false;
                    end
                else % forward direction
                    x = sequence(6,end);
                    % Ask the agent which action to perform
                    acta = possibleMoves(x, 0);
                    % Softmax action selection
                    %u = valueBasedDecision(Q(x,acta), R.decisionRule, R.beta, 0);
                    %u = acta(u);
                    % Random action selection
                    u = acta(randi(length(acta)));
                    % We observe the state in which this leads the agent
                    y = drand01(reshape(R.hatP(x,u,:),1,R.nS)); %argmax(R.hatP(x,u,:));
                    % Update Q
                    Qmax = max(Q,[],2);
                    if (replayMethod == 16) % DYNA thus MF update
                        RPE = R.hatR(x,u) + R.gamma * Qmax(y) - Q(x, u);
                        Q(x, u) = Q(x, u) + R.alpha * RPE;
                    else % MB thus MB update 
                        Q(x, u) = R.hatR(x, u) + R.gamma * sum(reshape(R.hatP(x, u, :), R.nS, 1) .* Qmax);
                    end
                    % we store the replayed element in the sequence
                    sequence = [sequence [x u R.hatR(x,u) 0 0 y 0]'];
                    iii = iii + 1;
                end
                % if it's a new sweep and we reach the central arm, we store the
                % characteristics of this sweep (L/R).
                if (newsweep&&(sequence(6,end)==24)) % old : state 24 then 26
                    newsweep = false;
                    % we store the side of the sweep
                    jjj = size(sequence,2);
                    % we loop until the first state found outside the central arm
                    while ((jjj > 1)&&((sequence(1,jjj)==15)||(sequence(1,jjj)==21)||(sequence(1,jjj)==22)||(sequence(1,jjj)==23)||(sequence(1,jjj)==24)||(sequence(1,jjj)==25)||(sequence(1,jjj)==26)||(sequence(1,jjj)==27))) %(sequence(1,jjj)==sequence(7,iii)))
                        jjj = jjj - 1;
                    end
                    sweepside = 24;
                    if ((sequence(1,jjj)<14)||(sequence(1,jjj)==18)||(sequence(1,jjj)==19))
                        % we are in the left arm
                        sweepside = 18;
                    end
                    if (sequence(1,jjj)>29)
                        % we are in the right arm
                        sweepside = 30;
                    end
                    %[jjj sequence(1,jjj)]
                    memorySweeps = [memorySweeps sequence(1,jjj)]; % 18, 30 or 24?
                end
                % if we arrive at decision-point, we initiate a new sweep
                if (~newsweep&&(sequence(6,end)==26))
                    newsweep = true;
                    %iii
                end
            end
            
        otherwise % incl. case 1 = basic VI
            sequence = zeros(7,R.window); % 54 states
            iii = 1;
            for x=1:R.nS
                for u=1:R.nA
                    % Update Q
                    Qmax = max(Q,[],2);
                    Q(x, u) = R.hatR(x,u) + R.gamma * sum(reshape(R.hatP(x,u,:),R.nS,1) .* Qmax);
                    sequence(:,iii) = [x u R.hatR(x,u) 0 0 argmax(R.hatP(x,u,:)) 0]';
                    iii = iii + 1;
                end
            end
    end
    
    % If the update is small enough (convergence), we stop
    variation = sum(sum(abs(Q2-Q)));
    if ((variation < R.replayiterthreshold)||((R.replaybudget > 0)&&(nbReplayCycle >= R.replaybudget)))
        quit = 1;
    end
end

% Computing the corresponding policy
[~, pol] = max(Q,[],2);
pol = pol';
