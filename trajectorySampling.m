function [Q, sequence, memorySweeps] = trajectorySampling(R, sequence, bufferRPE, memorySweeps, replayMethod, V, Q)
% This function performs trajectory sampling with a starting state and a
% Markovian model of the environment.
%
% INPUTS:
%
% OUTPUTS:
% 
%     created 20 May 2019
%     by Mehdi Khamassi
%     last modified 24 May 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 
    
    % strategy 1: trajectory sampling until visiting all states of
    % the environment

    % strategy 2: trajectory sampling until visiting all surprising
    % states in bufferRPE
    newsweep = false; % each time we go through decision point,
    % we consider that a new sweep starts.
    % each time we enter the central arm, newsweep is reset
    iii = 1;
    letsContinue = true;
    while letsContinue
        sequence = [sequence zeros(7,1)];
        % starting state
        if ((replayMethod < 18)&&(iii == 1))
            xxx = sequence(1,1); % not for combinations of prior sweep and traj sampl (18,19)
        else
            xxx = sequence(6,end-1);
        end
        if (~isempty(bufferRPE))
            bufferRPE(end,bufferRPE(1,:)==xxx) = 1; % we store having visited this state during traj sampl
        end
        % Ask the agent which action to perform
        acta = possibleMoves(xxx, 0);
        % we nevertheless don't want the agent to go back to its
        % previous position, but rather sample full trajectories
        for jjj=1:length(acta)
            if (argmax(R.hatP(xxx,acta(jjj),:))==sequence(1,end-1))
                acta(jjj) = 0;
            end
        end
        % we remove such positions, or re-compute acta if only
        % zeros
        if (sum(acta==0)==length(acta))
            acta = possibleMoves(xxx, 0);
        else
            acta(acta==0) = [];
        end
        % Softmax action selection
        %uuu = valueBasedDecision(Q(xxx,acta), R.decisionRule, R.betaReplay, 0);
        %uuu = acta(uuu);
        % Random action selection
        uuu = acta(randi(length(acta)));
        % We observe the state in which this leads the agent
        y = drand01(reshape(R.hatP(xxx,uuu,:),1,R.nS)); %argmax(R.hatP(x,u,:));
        % Update Q
        switch (replayMethod)
            case 15 % DYNA thus MF update
                Qmax = max(Q,[],2);
                RPE = R.hatR(xxx, uuu) + R.gamma * Qmax(y) - Q(xxx, uuu);
                Q(xxx, uuu) = Q(xxx, uuu) + R.alpha * RPE;
            case 18 % Update Q with PI
                Q(xxx, uuu) = R.hatR(xxx, uuu) + R.gamma * sum(reshape(R.hatP(xxx, uuu, :), R.nS, 1)' * V);
            otherwise % MB thus MB update
                Qmax = max(Q,[],2);
                Q(xxx, uuu) = R.hatR(xxx, uuu) + R.gamma * sum(reshape(R.hatP(xxx, uuu, :), R.nS, 1) .* Qmax);
        end
        % we store the replayed element in the sequence
        sequence(:,end) = [xxx uuu R.hatR(xxx,uuu) 0 0 y 0]';
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
        % deciding when to stop trajectory sampling
        switch (replayMethod)
            case {18,19} % when at least 1 surprising state has been visited, we stop
                % trajectory sampling
                if ((iii>=R.window)&&((isempty(bufferRPE))||((~isempty(bufferRPE))&&(sum(bufferRPE(end,:)==1)>=1))))
                    letsContinue = false;
                else
                    iii = iii + 1;
                end
            otherwise % fixed number R.window of iterations
                if (iii>=R.window)
                    letsContinue = false;
                else
                    iii = iii + 1;
                end
        end
    end
    
    % strategy 3: trajectory sampling until visiting all terminal
    % predessor states in bufferRPE after prioritized sweeping
end