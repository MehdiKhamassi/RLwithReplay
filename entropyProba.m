function [ result ] = entropyProba( proba )
% This function computes the entropy in a probability distribution
%
% INPUTS:
%     proba is a vector containing the probability of choosing each option
%     for a set of N options
%
% OUTPUTS:
%     result is a scalar indicating the entropy of this distribution
% 
%     created 1 Jun 2011
%     by Mehdi Khamassi
%     last modified 22 May 2012
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

    proba = proba / sum(proba);
    proba(proba==0) = 1e-10;
    result = max(-sum(proba.*log2(proba)),0);
end

