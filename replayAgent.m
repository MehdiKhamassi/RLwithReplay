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
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 
    
    %% Setting environment size based on MDP
    % Number of states available to the agent:
    nS = M.nS;
    % Number of actions available to the agent:
    nA = M.nA;
    
    %% Initialize model-free Q-learning (to decide which action to make)
    Q = ones(nS, nA) * 3; % Q-values
    RPEQ = zeros(1, nA); % delta for Q-learning
    alpha = 0.2; % Q-learning learning rate
    gamma = 0.99; % discount factor (time horizon for reward prediction)
    beta = 3; % exploration rate (inverse temperature)
    decisionRule = 'softmax'; % decision-rule

    %% Initialize model-based transition and reward functions
    hatP = ones(M.nS, M.nA, M.nS) / M.nS;
    hatR = zeros(M.nS, M.nA);
    N    = zeros(M.nS, M.nA);
    
    %% Initialize episodic memory
    window = 54; % size of window containing a number of iterations in episodic memory
    replayiterthreshold = 0.01; % threshold above which a cumulated change in Q requires another iteration of replay
    replaybudget = 10; % max nb of replay iterations allowed
    
    % Initialize the policy with random actions for all states
    pol = randi(M.nA,M.nS,1);
    
    % build a structure for the replay agent
    R = struct('nS', nS, 'nA', nA, 'alpha', alpha, 'beta', beta, 'gamma', gamma, 'decisionRule', decisionRule, 'Q', Q, 'RPEQ', RPEQ, 'pol', pol, 'hatP', hatP, 'hatR', hatR, 'N', N, 'window', window, 'replayiterthreshold', replayiterthreshold, 'replaybudget', replaybudget');

end