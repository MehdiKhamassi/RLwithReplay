function M = MDP()
% This functions returns the Markov Decision Process (MDP) model for the
% multiple-T-maze task of the group of A. David Redish.
%
% OUTPUT: M is a structure containing the MDP
%
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

% Action encoding
N = 1;
S = 2;
E = 3;
W = 4;

% Number of states of the robot:
nS = 54;
% accessible states
stata = [1 2 3 4 5 6 7 12 13 15 18 19 21 22 23 24 25 26 27 30 31 36 37 42 43 48 49 50 51 52 53 54]; % if double T-maze
%stata = [1 2 3 4 5 6 7 12 13 18 19 21 22 23 24 25 26 27 30 31 36 37 42 43 48 49 50 51 52 53 54]; % if 8-maze

% Number of actions available to the robot:
nA = 4;

% stochasticity of the reward
stochastic = true;

% Transition probabilities for action N. Lines correspond to the "current"
% state and columns to the "next" state.
% For example, P(2, N, 1) represents the probability of moving to state 1
% from state 2 when choosing action N.
P(:, N, :) = zeros(nS, nS);
for iii = 1:nS
    switch (iii)
        case {2,3,4,5,6,22,23,24,26,27,50,51,52,53,54}
            P(iii, N, iii-1) = 1;
        otherwise
            P(iii, N, iii) = 1;
    end
end

% Transition probabilities for action S. Lines correspond to the "current"
% state and columns to the "next" state.
P(:, S, :) = zeros(nS, nS);
for iii = 1:nS
    switch (iii)
        case {1,2,3,4,5,21,22,23,25,26,49,50,51,52,53}
            P(iii, S, iii+1) = 1;
        otherwise
            P(iii, S, iii) = 1;
    end
end

% Transition probabilities for action E. Lines correspond to the "current"
% state and columns to the "next" state.
P(:, E, :) = zeros(nS, nS);
for iii = 1:nS
    switch (iii)
        case {1,6,7,12,13,15,18,19,21,24,25,30,31,36,37,42,43,48}
            P(iii, E, iii+6) = 1;
        otherwise
            P(iii, E, iii) = 1;
    end
end

% Transition probabilities for action W. Lines correspond to the "current"
% state and columns to the "next" state.
P(:, W, :) = zeros(nS, nS);
for iii = 1:nS
    switch (iii)
        case {7,12,13,18,19,21,24,25,27,30,31,36,37,42,43,48,49,54} % add 21 if multiple T-maze
            P(iii, W, iii-6) = 1;
        otherwise
            P(iii, W, iii) = 1;
    end
end

% Reward function, reprensented as a state x action matrix. r(2, N) represents
% the reward of choosing action N in state 2.
r = zeros(nS, nA);
if (stochastic)
    r(5,S) = 0.75;
    r(53,S) = 0.25;
else
    r(5,S) = 1; % rule shift: r(53,S) = 1;
end

% defining the parameters of the task (default values)
totalDuration = 5000; % total duration of the experiment
conditionDuration = 2000; % after which we change condition
constraint = 1; % 1 only forward moves, 0 no wall bump, -1 no constraint
departureState = 25; % departure state
replayPosition = stata; % 1:54; % in all states or rwd locations only: [6 54]; % or departure state only: 25; % state(s) where replays are allowed
logSequenceLength = 2000; % max length of stored replay sequences

% Distribution of the states where replay can occur.
P0 = zeros(1,nS);
P0(replayPosition) = 1 / length(replayPosition);

% We now create a structure that stores all the elements of the MDP
M = struct('nS', nS, 'nA', nA, 'P', P, 'r', r, 'P0', P0, 'totalDuration', totalDuration, 'conditionDuration', conditionDuration, 'constraint', constraint, 'departureState', departureState, 'replayPosition', replayPosition, 'logSequenceLength', logSequenceLength, 'stochastic', stochastic, 'stata', stata);
