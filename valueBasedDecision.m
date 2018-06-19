function [ Y, proba ] = valueBasedDecision( values, decisionRule, beta, bias )
% This function computes the probability distribution over possible options
% (or "actions" here) based on their learned Q-values, and then selects one
% of these options based on this probability distribution
% In this version of valueBasedDecision, we can have N controllers/systems,
% whose respective option values are weighted by a controller-specific beta
% parameter.
%
% INPUTS:
%     values contains N lines (1 line per controller), 1 column for each competing option
%     values contains only option values in the current state
%     decisionRule can either be 'matching', 'max', 'softmax' or 'espilon'
%     beta (>= 0) contains 1 line, and N columns (thus one parameter per controller)
%     bias is of the same size as values: bias for controller 1 for option
%     1, for option 2, ... bias is meant to test offsetting effects on
%     action selection. Nevertheless, it is most of the time set to 0.
%
% OUTPUTS:
%     Y is a natural number identifying the chosen option
%     proba is a vector containing the probability of choosing each option
%     for a set of M options
% 
%     created 5 Apr 2011
%     by Mehdi Khamassi
%     last modified 3 Feb 2015
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

    
    nbC = size(values,1); % nb controllers
    nbA = size(values,2); % nb actions
    
    switch (decisionRule),
        case 'matching',
            if (sum(beta) ~= 0),
                proba = (beta*values) / sum(beta);
            else,
                proba = ones(1,nbA) / nbC;
            end;
            Y = drand01(proba); % rolls a dice and chooses an action depending on its proba
        case 'max',
            if (sum(beta) ~= 0),
                proba = (beta*values) / sum(beta);
            else,
                proba = ones(1,nbA) / nbC;
            end;
            Y = argmax(proba);
        case 'softmax',
            combVal = min(beta*values+bias,ones(1,nbA)*700); % not more than 700 due to exponential
            proba = exp(combVal) / sum(exp(combVal));
            Y = drand01(proba); % rolls a dice and chooses an action depending on its proba
        case 'epsilon'
            proba = (ones(1,nbA) * beta + bias) / (nbA - 1); % beta == epsilon
            [~, idxBest] = max(values+bias);
            proba(idxBest) = bias(idxBest)/ (nbA - 1) + 1 - beta; 
            Y = drand01(proba); % rolls a dice and chooses an action depending on its proba
    end;
    proba = max(proba,ones(1,nbA)*1e-100);
end

