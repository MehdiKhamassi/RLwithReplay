function [ TDerror, values ] = temporalDifferenceError( reward, minRwd, maxRwd, action, values, newvalues, weight, alpha, gamma, kappa1, kappa2, alpha2, Qinit )
% This function updates Q-values through temporal-difference learning based on the reward signal received from the environment.
% In this version of temporalDifferenceError, a generalized reinforcement learning algorithm is used (as in Khamassi et al 2015
% Cerebral Cortex) so that unchosen actions also see their values updated according to a forgetting mechanism with forgetting
% rate alpha2. Forgetting pulls Q-values towards Qinit. Finally, as in Ito & Doya 2009 J Neurosci, the function controls the
% impact of reward (kappa1, which should be equal to 1 by default) and the impact of no-reward (kappa2, which should be equal
% to 0 by default).
%
% INPUTS:
% reward = reward value obtained by the agent
% minRwd = minimum possible reward in the task
% maxRwd = maximum possible reward in the task
% action = action that was performed
% values = action values in the previous state
% newvalues = action values in the new state
% weight = probability of having being in the previous state (POMDP)
% alpha = learning rate
% gamma = discount factor
% kappa1 = magnitude of reward
% kappa2 = magnitude of no-reward 
%
% OUTPUTS:
% TDerror is the reward prediction error per action (vector)
% values is a vector containing the updated values
% 
%     created 5 Apr 2011
%     by Mehdi Khamassi
%     last modified 16 Apr 2013
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

    kapa3 = 0; % before kapa3 = kapa2 with kapa2 always equal to 0
    % alpha2 = forgetting rate for non chosen actions
    nbAction = length(values);
    TDerror = zeros(1,nbAction);
    TDerror(action) = (reward~=0)*reward*kappa1 + (reward==0)*(reward-1)*kappa2 + gamma*max(newvalues) - values(action);
    values(action) = values(action) + weight*alpha*TDerror(action);
    if (nbAction > 1),
        for iii=0:(nbAction-2), % we update the values of non-chosen actions
            TDerror(mod(action+iii,nbAction)+1) = Qinit + (reward>minRwd)*minRwd*kapa3 + (reward<=minRwd)*maxRwd*kapa3 + gamma*max(newvalues) - values(mod(action+iii,nbAction)+1);
            values(mod(action+iii,nbAction)+1) = values(mod(action+iii,nbAction)+1) + weight*alpha2*TDerror(mod(action+iii,nbAction)+1);
        end;
    end;
end

