function [R, logs] = simulateRLwithReplay(M, R, replayMethod)
% This function simulates a full RL+replay experiment, choosing an action
% at each timestep, observing consequence (reward and new state), learning
% the Q-values of both MB (Q_MB) and MF (Q_MF) systems, and trying to
% perform one or more replay cycles before choosing a novel action at the
% next timestep
%
% INPUTS:
%     M is a structure containing the MDP
%     R is a structure containing the learning agent
%     replayMethod is the chosen method for replays
%
% OUTPUTS:
%     R is the structure containing the updated agent
%     logs is a structure storing the sequences of states replayed during
%     each replay cycle
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 24 May 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

% Action encoding
N = 1;
S = 2;
E = 3;
W = 4;

% Initialize the buffer where we store sweep characteristics
logs.memorySweeps = []; % we store L or R for each sweep
logs.memorySide = []; % we store L or R for each state replayed
logs.replaySequence = []; % we store the full sequence of replay
logs.sequence = zeros(7,M.totalDuration); % state action reward replay-duration rule newstate replay-cycle
bufferRPE = []; % we store RPEs during the trial to order the replay buffer

% Draw an initial state
x = M.departureState; % drand01(M.P0); % randi(M.nS, 1, 1);

% Run the experiment
letsContinue = true;
iter = 1;
while (letsContinue)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% task rule shift (reward position change) %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (mod(iter,M.conditionDuration) == 0)
        % SHORT EXPERIMENT, one task rule change
        % (or uncomment lines 35,36 for repeated task rule changes
        % every M.conditionDuration steps)
        if (M.r(5,S) == 1) % reward currently in state 5
            M.r(5,S) = 0;
            M.r(53,S) = 1; % comment if extinction experiment!
        else % reward currently in state 53
            %M.r(53,S) = 0;
            %M.r(5,S) = 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    %% decision-making %%
    %%%%%%%%%%%%%%%%%%%%%
    % Draw a random action and get the next state and reward
    %u = randi(M.nS, 1, 1);
    % Rather ask the agent which action to perform
    switch (M.constraint)
        case 1 % the agent can only move forward
            acta = possibleMoves(x, 1);
            u = valueBasedDecision(R.Q(x,acta), R.decisionRule, R.beta, 0);
            u = acta(u);
        case 0 % the agent cannot bump into the walls
            acta = possibleMoves(x, 0);
            u = valueBasedDecision(R.Q(x,acta), R.decisionRule, R.beta, 0);
            u = acta(u);
        otherwise % the agent can do whatever
            u = valueBasedDecision(R.Q(x,:), R.decisionRule, R.beta, 0);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% observing consequence in the environment %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (M.stochastic) % stochastic reward for the long experiment
        y = drand01(reshape(M.P(x, u, :), M.nS, 1)');
        % probabilistic reward
        r = drand01([(1 - M.r(x, u)) M.r(x, u)]) - 1;
    else
        [y, r] = MDPStep(M, x, u);
    end
%     if (r > 0)
%         [iter x u y r] % logs
%     end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% model-based learning %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update the number of visits for the current state-action pair
    R.N(x, u) = R.N(x, u) + 1;
    % Update transition matrix (stochastic)
    R.hatP(x, u, :) = (1 - 1/R.N(x, u)) * R.hatP(x, u, :) + reshape((1:R.nS == y) / R.N(x, u), 1, 1, R.nS);
    if (M.stochastic)
        % Update reward function (stochastic)
        R.hatR(x, u) = (1 - 1/R.N(x, u)) * R.hatR(x, u) + r / R.N(x, u);
    else
        % Update reward function (deterministic)
        R.hatR(x, u) = r;
    end
    % good old simple RTDP method (Sutton 1990) to update Q
    switch (replayMethod)
        case {2,18} % PI methods
            pol = R.pol;
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
%             % Performing PI update
%             for xxx=1:R.nS
%                 for uuu=1:R.nA
%                     Q(xxx, uuu) = R.hatR(xxx, uuu) + R.gamma * sum(reshape(R.hatP(xxx, uuu, :), R.nS, 1)' * V);
%                 end
%             end
            newQ = R.hatR(x, u) + R.gamma * sum(reshape(R.hatP(x, u, :), R.nS, 1)' * V);
        otherwise % VI, MF or Dyna methods
            Qmax = max(R.Q, [], 2);
            newQ = R.hatR(x, u) + R.gamma * sum(reshape(R.hatP(x, u, :), R.nS, 1) .* Qmax);
    end
    RPE_MB = newQ - R.Q(x, u);
    Q_MB = newQ;
    switch (replayMethod)
        case {1,2,6,8,10,11,12,13,18,19,20} % MB-RL methods
            R.Q(x, u) = Q_MB;
            RPE = RPE_MB;
        case {0,3,4,5,7,9} % MF-RL methods
        case {14,15,16,17} % Dyna-RL methods
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %% model-free learning %%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % Update Q in a model-free manner
    [ RPE_MF, Q_MF ] = temporalDifferenceError( r, 0, 1, u, R.Q(x,:), R.Q(y,:), 1, R.alpha, R.gamma, 1, 0, 0, 0 );
    switch (replayMethod)
        case {1,2,6,8,10,11,12,13,18,19,20} % MB-RL methods
        case {0,3,4,5,7,9} % MF-RL methods
            R.Q(x,:) = Q_MF;
            RPE = RPE_MF(u);
        case {14,15,16,17} % Dyna-RL methods
            R.Q(x,:) = Q_MF;
            RPE = RPE_MF(u);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% storing the step in episodic memory %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    logs.sequence(1, iter) = x; % storing the current state
    if (M.r(5,S) >= M.r(53,S)) % reward currently in state 5
        logs.sequence(5, iter) = 5; % storing the current task rule
    else % reward currently in state 53
        logs.sequence(5, iter) = 53; % storing the current task rule
    end
    logs.sequence(2, iter) = u; % storing the action
    logs.sequence(3, iter) = r; % storing the reward
    logs.sequence(6, iter) = y; % storing the arrival state
    % we add the (x,RPE(u)) to bufferRPE only if x is not already in the buffer
    switch (replayMethod)
        case 20 % (state,action)-based prioritized sweeping
            if ((size(bufferRPE,2) == 0)||(sum(bufferRPE(1,:)==x&bufferRPE(2,:)==u)==0))
                if (abs(RPE) > R.replayiterthreshold) % high priority
                    bufferRPE = [bufferRPE [logs.sequence(1:3,iter) ; RPE ; abs(RPE) ; logs.sequence(6:7,iter)]];
                end
            else % y is already in bufferRPE
                if (abs(RPE) > bufferRPE(5,bufferRPE(1,:)==x&bufferRPE(2,:)==u)) % higher priority
                    bufferRPE(:,bufferRPE(1,:)==x&bufferRPE(2,:)==u) = [logs.sequence(1:3,iter) ; RPE ; abs(RPE) ; logs.sequence(6:7,iter)];
                end
            end         
        otherwise
           if ((size(bufferRPE,2) == 0)||(sum(bufferRPE(1,:)==x)==0))
                if (abs(RPE) > R.replayiterthreshold) % high priority
                    bufferRPE = [bufferRPE [logs.sequence(1:3,iter) ; RPE ; abs(RPE) ; logs.sequence(6:7,iter)]];
                end
            else % y is already in bufferRPE
                if (abs(RPE) > bufferRPE(5,bufferRPE(1,:)==x)) % higher priority
                    bufferRPE(:,bufferRPE(1,:)==x) = [logs.sequence(1:3,iter) ; RPE ; abs(RPE) ; logs.sequence(6:7,iter)];
                end
            end 
    end
    
    % we search for predecessors of x
    if (((replayMethod == 6)||(replayMethod == 11)||(replayMethod == 12)||(replayMethod == 17)||(replayMethod == 18)||(replayMethod == 19)||(replayMethod == 20))&&(abs(RPE) > R.replayiterthreshold)) % only for MB-prior methods
        for aaa = 1:M.nA
            pred = R.hatP(:,aaa,x)>(1/M.nS);
            indexes = (1:M.nS);
            pred = indexes(pred);
            while (~isempty(pred))
                % compute RPE for each pred
                switch (replayMethod)
                    case {2,18} % PI methods
                        Qpred = R.hatR(pred(1),aaa) + R.gamma * sum(reshape(R.hatP(x, aaa, :), R.nS, 1)' * V);
                        RPEpred = Qpred - R.Q(pred(1),aaa);
                    case {6,11,12,19,20} % VI methods
                        Qmax = max(R.Q, [], 2);
                        Qpred = R.hatR(x, u) + R.gamma * sum(reshape(R.hatP(x, u, :), R.nS, 1) .* Qmax);
                        RPEpred = Qpred - R.Q(pred(1),aaa);
                    otherwise % MF or Dyna methods
                        Qmax = max(R.Q, [], 2);
                        RPEpred = R.hatR(pred(1),aaa) + R.gamma * Qmax(pred(1)) - R.Q(x, aaa);
                end
                % add (pred,RPEpred) to bufferRPE
                switch (replayMethod)
                    case 20 % (state,action)-based prioritized sweeping
                        if (sum(bufferRPE(1,:)==pred(1)&bufferRPE(2,:)==aaa)==0) % predecessor not already in bufferRPE
                            if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > R.replayiterthreshold) % high priority
                                bufferRPE = [bufferRPE [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0]];
                            end
                        else % predecessor already in bufferRPE
                            if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > bufferRPE(5,bufferRPE(1,:)==pred(1))) % higher priority than previously stored in buffer for this predecessor
                                bufferRPE(:,bufferRPE(1,:)==pred(1)&bufferRPE(2,:)==aaa) = [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0];
                            end
                        end
                    otherwise % state-based prioritized sweeping
                        if (sum(bufferRPE(1,:)==pred(1))==0) % predecessor not already in bufferRPE
                            if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > R.replayiterthreshold) % high priority
                                if (replayMethod == 12)
                                    bufferRPE = [bufferRPE [pred(1) ; aaa ; 0 ; R.gamma * R.hatP(pred(1),aaa,x) * RPEpred ; R.gamma * R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0]];
                                else
                                    bufferRPE = [bufferRPE [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0]];
                                end
                            end
                        else
                            if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > bufferRPE(5,bufferRPE(1,:)==pred(1))) % higher priority than previously stored in buffer for this predecessor
                                if (replayMethod == 12)
                                    bufferRPE(:,bufferRPE(1,:)==pred(1)) = [pred(1) ; aaa ; 0 ; R.gamma * R.hatP(pred(1),aaa,x) * RPEpred ; R.gamma * R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0];
                                else
                                    bufferRPE(:,bufferRPE(1,:)==pred(1)) = [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0];
                                end
                            end
                        end
                end
                pred(1) = [];
            end
        end
    end
    
    %%%%%%%%%%%%%
    %% replays %%
    %%%%%%%%%%%%%
    % replay only at possible replay state(s)
    if (M.P0(y) > 0)
        switch (replayMethod)
            case 1 %% VI as replay method
                [Q, pol, nbReplayCycle] = VI(M, R, [], [], replayMethod);
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = nbReplayCycle * R.nS * R.nA; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                
            case 2 %% PI as replay method
                [Q, pol, nbReplayCycle] = PI(M, R, [], [], replayMethod);
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = nbReplayCycle * R.nS * R.nA; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
        
            %% MF episodeReplay as replay methods
            case 3 % MF forward replay of a sequence in a time window
                replaybuffer = logs.sequence(:,max(1,iter-R.window+1):iter);
                [~, start] = max(replaybuffer(1,:)==y);
                [Q, pol, nbReplayCycle, duration, buffersweeps, bufferside] = MFepisodeReplay(R, replaybuffer(:,start:end));
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = duration; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                durationReplaySequence = size(replaybuffer(1,:),2);
                logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence ones(1,R.window-3)] ; [replaybuffer(1,:) ones(1,R.window-durationReplaySequence)*-1]]; % we store the full sequence of replay
        
            case {4,5} % MF prioritized sweeping
                if (~isempty(bufferRPE))
                    replaybuffer = bufferRPE(:,max(1,end-R.window+1):end);
                    [~, index] = sort(replaybuffer(5,:),'descend');
                    replaybuffer = replaybuffer(:,index);
                    start = 1;
                    [Q, pol, nbReplayCycle, duration, buffersweeps, bufferside] = MFepisodeReplay(R, replaybuffer(:,start:end));
                    if (replayMethod == 4)
                        bufferRPE = []; % we reset the buffer of RPEs
                    end
                    % we store the result of the replay
                    R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                    logs.sequence(4,iter) = duration; % we store replay duration
                    logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                    logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                    logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                    durationReplaySequence = size(replaybuffer(1,:),2);
                    logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence 1 ones(1,R.window-4)] ; [replaybuffer(1,:) ones(1,R.window-durationReplaySequence)*-1]]; % we store the full sequence of replay
                else
                    logs.sequence(4,iter) = 0; % we store replay duration
                    logs.sequence(7,iter) = 0; % we store replay cycles
                end
                
            case 7 % MF Shuffled replay
                replaybuffer = logs.sequence(:,max(1,iter-R.window+1):iter);
                index = randperm(size(replaybuffer,2));
                replaybuffer = replaybuffer(:,index);
                start = 1;
                [Q, pol, nbReplayCycle, duration, buffersweeps, bufferside] = MFepisodeReplay(R, replaybuffer(:,start:end));
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = duration; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                durationReplaySequence = size(replaybuffer(1,:),2);
                logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence ones(1,R.window-3)] ; [replaybuffer(1,:) ones(1,R.window-durationReplaySequence)*-1]]; % we store the full sequence of replay
            
            case 9 % MF Backward replay (Lin 1992)
                replaybuffer = logs.sequence(:,max(1,iter-R.window+1):iter);
                replaybuffer = replaybuffer(:,end:-1:1); % reversing order
                start = 1;
                [Q, pol, nbReplayCycle, duration, buffersweeps, bufferside] = MFepisodeReplay(R, replaybuffer(:,start:end));
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = duration; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                durationReplaySequence = size(replaybuffer(1,:),2);
                logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence ones(1,R.window-3)] ; [replaybuffer(1,:) ones(1,R.window-durationReplaySequence)*-1]]; % we store the full sequence of replay
                
            %% MB/DYNA inference as replay methods
            case {6,11,12,17,20} % MB/DYNA prioritized sweeping
                if (~isempty(bufferRPE))
                    replaybuffer = bufferRPE(:,max(1,end-R.window+1):end);
                    [~, index] = sort(replaybuffer(5,:),'descend');
                    replaybuffer = replaybuffer(:,index);
                    start = 1;
                    trajectbuffer = logs.sequence(:,iter);
                    [Q, pol, nbReplayCycle, ~, buffersweeps, bufferside, replaybuffer, bufferRPE] = VI(M, R, replaybuffer(:,start:end), trajectbuffer(:,start:end), replayMethod);
                    if (replayMethod == 6)
                        bufferRPE = []; % we reset the buffer of RPEs
                    end
                    % we store the result of the replay
                    R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                    logs.sequence(4,iter) = size(replaybuffer(1,:),2); % we store replay duration
                    logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                    logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                    logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                    durationReplaySequence = size(replaybuffer(1,:),2);
                    logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence 1 ones(1,M.logSequenceLength-4)] ; [replaybuffer(1,1:min(M.logSequenceLength,durationReplaySequence)) ones(1,M.logSequenceLength-durationReplaySequence)*-1]]; % we store the full sequence of replay
                else
                    logs.sequence(4,iter) = 0; % we store replay duration
                    logs.sequence(7,iter) = 0; % we store replay cycles
                end
                %logs.sequence(:,iter)'
                
            case {8,14} %% MB-RL/Dyna-RL shuffled: VI with shuffled state
                replaybuffer = logs.sequence(:,iter);
                start = 1;
                [Q, pol, nbReplayCycle, ~, buffersweeps, bufferside, replaybuffer] = VI(M, R, replaybuffer(:,start:end), replaybuffer(:,start:end), replayMethod);
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = size(replaybuffer(1,:),2)-1; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                durationReplaySequence = size(replaybuffer(1,:),2)-1;
                logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence ones(1,M.logSequenceLength-3)] ; [replaybuffer(1,2:min(M.logSequenceLength+1,durationReplaySequence)) ones(1,M.logSequenceLength+1-durationReplaySequence)*-1]]; % we store the full sequence of replay
                
            case {10,15} % MB/DYNA trajectory sampling
                replaybuffer = logs.sequence(:,iter);
                start = 1;
                [Q, pol, nbReplayCycle, ~, buffersweeps, bufferside, replaybuffer] = VI(M, R, [], replaybuffer(:,start:end), replayMethod);
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = size(replaybuffer(1,:),2)-1; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                durationReplaySequence = size(replaybuffer(1,:),2)-1;
                logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence ones(1,M.logSequenceLength-3)] ; [replaybuffer(1,2:min(M.logSequenceLength+1,durationReplaySequence)) ones(1,max(0,M.logSequenceLength+1-durationReplaySequence))*-1]]; % we store the full sequence of replay
                %logs.sequence(:,iter)'
                
            case {13,16} % MB/DYNA bidirectional planning
                iii = argmax(logs.sequence(3,:)); % locate reward (i.e. goal)
                if false %((iii == 1)&&(logs.sequence(3,iii) == 0))
                    iii = iter;
                end
                replaybuffer = logs.sequence(:,iii);
                replaybuffer(6,1) = logs.sequence(1,iter); % store current position
                start = 1;
                [Q, pol, nbReplayCycle, ~, buffersweeps, bufferside, replaybuffer] = VI(M, R, [], replaybuffer(:,start:end), replayMethod);
                % we store the result of the replay
                R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                logs.sequence(4,iter) = size(replaybuffer(1,:),2)-1; % we store replay duration
                logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                durationReplaySequence = size(replaybuffer(1,:),2)-1;
                logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence ones(1,M.logSequenceLength-3)] ; [replaybuffer(1,2:min(M.logSequenceLength+1,durationReplaySequence)) ones(1,max(0,M.logSequenceLength+1-durationReplaySequence))*-1]]; % we store the full sequence of replay
                
            %% MB inference as replay method combined with policy iteration
            case 18 % MB prioritized sweeping with policy iteration
                if (~isempty(bufferRPE))
                    replaybuffer = bufferRPE(:,max(1,end-R.window+1):end);
                    [~, index] = sort(replaybuffer(5,:),'descend');
                    replaybuffer = replaybuffer(:,index);
                    start = 1;
                    trajectbuffer = logs.sequence(:,iter);
                    [Q, pol, nbReplayCycle, ~, buffersweeps, bufferside, replaybuffer, bufferRPE] = PI(M, R, replaybuffer(:,start:end), trajectbuffer(:,start:end), replayMethod);
                    % we store the result of the replay
                    R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                    logs.sequence(4,iter) = size(replaybuffer(1,:),2); % we store replay duration
                    logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                    logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                    logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                    durationReplaySequence = size(replaybuffer(1,:),2);
                    logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence 1 ones(1,M.logSequenceLength-4)] ; [replaybuffer(1,1:min(M.logSequenceLength,durationReplaySequence)) ones(1,M.logSequenceLength-durationReplaySequence)*-1]]; % we store the full sequence of replay
                else
                    logs.sequence(4,iter) = 0; % we store replay duration
                    logs.sequence(7,iter) = 0; % we store replay cycles
                end
                
            %% MB inference as replay method
            case 19 % MB prioritized sweeping + traj sampling until surprising states
                if (~isempty(bufferRPE))
                    replaybuffer = bufferRPE(:,max(1,end-R.window+1):end);
                    [~, index] = sort(replaybuffer(5,:),'descend');
                    replaybuffer = replaybuffer(:,index);
                else
                    replaybuffer = [];
                end
                    start = 1;
                    trajectbuffer = logs.sequence(:,iter);
                    [Q, pol, nbReplayCycle, ~, buffersweeps, bufferside, replaybuffer, bufferRPE] = VI(M, R, replaybuffer, trajectbuffer(:,start:end), replayMethod);
                    % we store the result of the replay
                    R.Q = Q; R.pol = pol; % we store in R the result of VI/PI
                    logs.sequence(4,iter) = size(replaybuffer(1,:),2); % we store replay duration
                    logs.sequence(7,iter) = nbReplayCycle; % we store replay cycles
                    logs.memorySweeps = [logs.memorySweeps ; [iter sum(buffersweeps==18) sum(buffersweeps==30) length(buffersweeps)]];
                    logs.memorySide = [logs.memorySide ; [iter sum(bufferside==18) sum(bufferside==30) length(bufferside)]];
                    durationReplaySequence = size(replaybuffer(1,:),2);
                    logs.replaySequence = [logs.replaySequence ; [iter nbReplayCycle durationReplaySequence 1 ones(1,M.logSequenceLength-4)] ; [replaybuffer(1,1:min(M.logSequenceLength,durationReplaySequence)) ones(1,M.logSequenceLength-durationReplaySequence)*-1]]; % we store the full sequence of replay
                    %logs.sequence(:,iter)'
%                 else
%                     logs.sequence(4,iter) = 0; % we store replay duration
%                     logs.sequence(7,iter) = 0; % we store replay cycles
%                 end
                
            otherwise % no replay
                logs.sequence(4,iter) = 0; % we store replay duration
                logs.sequence(7,iter) = 0; % we store replay cycles
        end % end of switch(replayMethod)
    else
        logs.sequence(4,iter) = 0; % null replay duration
        logs.sequence(7,iter) = 0; % null replay cycle
    end
    
    % The next state becomes the current state
    x = y;
    iter = iter + 1;
    
    % check whether the experiment stops
    if ((iter >= M.totalDuration)&&(y == M.departureState))
        letsContinue = false;
    end
end

% Compute a policy based on the estimate of the optimal Q-function
pol = zeros(R.nS, 1);
for x=1:R.nS
    pol(x) = argmax(R.Q(x, :));
end

