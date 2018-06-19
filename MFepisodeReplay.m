function [Q, pol, nbReplayCycle, duration, memorySweeps, memorySide] = MFepisodeReplay(R, sequence)
% This function computes the optimal Q-function and the corresponding policy
% for the MDP M using model-free reinforcement learning
% during replay of episodic memory (~ Lin 1992)
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
%     memorySweeps stores for each replay cycle the side of the maze where
%     the sweep of states finished when reaching back the central arm
%     memorySide stores for each replay cycle the proportion of states that
%     were replayed either on the left or on the right arm
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

%% Initialization
% % Initialize Q as a matrix of state x action
% Q       = zeros(R.nS, R.nA);
Q = R.Q;

% Initialize the buffer where we store sweep characteristics
memorySweeps = []; % we store L or R for each sweep
memorySide = []; % we store L or R for each state replayed

% Episode replay loop
quit = 0;
nbReplayCycle = 0;
duration = 0;
while (~quit)
    nbReplayCycle = nbReplayCycle + 1;
    duration = duration + size(sequence,2);

    % storing the current Q function
    Q2 = Q;
    
    % Update the Q function with replay
    newsweep = false; % each time we go through decision point,
    % we consider that a new sweep starts.
    % each time we enter the central arm, newsweep is reset
    for iii=1:size(sequence,2)
        %[iii sequence(:,iii)']
        % Update Q in a model-free manner
        [ ~, Qval ] = temporalDifferenceError( sequence(3,iii), 0, 1, sequence(2,iii), Q(sequence(1,iii),:), Q(sequence(6,iii),:), 1, R.alpha, R.gamma, 1, 0, 0, 0 );
        % Store the new Q-values in R's Q-function
        Q(sequence(1,iii),:) = Qval;
        % if it's a new sweep and we reach the central arm, we store the
        % characteristics of this sweep (L/R).
        if (newsweep&&(sequence(6,iii)==24))
            newsweep = false;
            % we store the side of the sweep
            jjj = iii;
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
        if (~newsweep&&(sequence(6,iii)==26))
            newsweep = true;
        end
        % we now store the side of each replayed state
        if ((sequence(1,iii)<14)||(sequence(1,iii)==18)||(sequence(1,iii)==19))
            % we are in the left arm
            sweepside = 18;
            memorySide = [memorySide sweepside]; % 18, 30 or 24?
        else
            if (sequence(1,iii)>29)
                % we are in the right arm
                sweepside = 30;
                memorySide = [memorySide sweepside]; % 18, 30 or 24?
            else
                sweepside = 24;
                memorySide = [memorySide sweepside]; % 18, 30 or 24?
            end
        end
    end;
    
    % If the update of Q-values is small enough (convergence), we do not
    % start a new replay cycle
    variation = sum(sum(abs(Q2-Q)));
    if ((variation < R.replayiterthreshold)||((R.replaybudget > 0)&&(nbReplayCycle >= R.replaybudget)))
        quit = 1;
    end
end

% Compute the corresponding policy
[~, pol] = max(Q,[],2);
pol = pol';