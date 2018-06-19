function [y, r] = MDPStep(M, s, a)
% This function executes a step on the MDP M given current state s and action a.
% It returns a next state y and a reward r
%
% INPUTS:
%     M is a structure containing the MDP
%     s is the current state
%     a is the performed action
%
% OUTPUTS:
%     y is the resulting state
%     r is the resulting reward
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

y = drand01(reshape(M.P(s, a, :), M.nS, 1)');
r = M.r(s, a);