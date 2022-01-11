function R = replayAgent(M)
% This function creates the learning agents combining model-based and
% model-free RL in parallel, one of these two types of learning or both
% will be used depending on the chosen replayMethod (chosen in main.m)
%
% INPUT: M is a structure containing the MDP
%
% OUTPUT: R is the structure containing the agent
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 3 December 2021
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 
    
    %% Setting environment size based on MDP
    % Number of states available to the agent:
    nS = M.nS;
    % Number of actions available to the agent:
    nA = M.nA;
    
    %% Initialize model-free Q-learning (to decide which action to make)
    Q = zeros(nS, nA); % Q-values
    RPEQ = zeros(1, nA); % delta for Q-learning
    alpha = 0.2; % Q-learning learning rate
    gamma = 0.99; % discount factor (time horizon for reward prediction)
    beta = 3; %10 for PI versus 3 for VI; % exploration rate (inverse temperature)
    betaReplay = 10; % for exploration during trajectory sampling
    decisionRule = 'softmax'; % decision-rule

    %% Initialize model-based transition and reward functions
    hatP = ones(nS, nA, nS) / nS;
    hatR = zeros(nS, nA);
    N    = zeros(nS, nA);
    
    %% Initialize episodic memory
    window = 10; %54; % size of window containing a number of iterations in episodic memory
    replayiterthreshold = 0.001; % threshold above which a cumulated change in Q requires another iteration of replay
    replaybudget = 10; % max nb of replay iterations allowed
    
    % Initialize the policy with random actions for all states
    pol = randi(nA, nS, 1);
    
    % build a structure for the replay agent
    R = struct('nS', nS, 'nA', nA, 'alpha', alpha, 'beta', beta, 'betaReplay', betaReplay, 'gamma', gamma, 'decisionRule', decisionRule, 'Q', Q, 'RPEQ', RPEQ, 'pol', pol, 'hatP', hatP, 'hatR', hatR, 'N', N, 'window', window, 'replayiterthreshold', replayiterthreshold, 'replaybudget', replaybudget');

end