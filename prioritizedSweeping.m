function [Q, sequence, bufferRPE] = prioritizedSweeping(R, sequence, bufferRPE, replayMethod, V, Q)
% This function performs prioritized sweeping with a
% Markovian model of the environment, and a memory buffer containing all
% surprising events in episodic memory, associated to the amount of
% surprise, that is, reward prediction error
%
% INPUTS:
%     R is the replay agent
%     V is the currently estimated value function
%     Q is the currently estimated (state,action) value function
%     bufferRPE contains the list of elements to be replayed, ordered by descending abs(RPE)
%     sequence contains the sampled replay trajectory
%
% OUTPUTS:
% 
%     created 20 May 2019
%     by Mehdi Khamassi
%     last modified 24 May 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 
    
    iii = 0;
    letsContinue = true;
    while letsContinue
        element = bufferRPE(:,1);
        bufferRPE(:,1) = [];
        x = element(1);
        u = element(2);
        % Update Q
        Qmax = max(Q,[],2);
        if (replayMethod == 17) % DYNA thus MF update
            y = drand01(reshape(R.hatP(x,u,:),1,R.nS));
            RPE = R.hatR(x,u) + R.gamma * Qmax(y) - Q(x, u);
            newQ = Q(x, u) + R.alpha * RPE;
            Q(x, u) = newQ;
            newElement = [x ; u ; R.hatR(x,u) ; RPE ; abs(RPE) ; y ; 0];
        else % MB thus MB update
            if (replayMethod == 18) % PI
                newQ = R.hatR(x, u) + R.gamma * sum(reshape(R.hatP(x, u, :), R.nS, 1)' * V);
            else % VI
                newQ = R.hatR(x,u) + R.gamma * sum(reshape(R.hatP(x,u,:),R.nS,1) .* Qmax);
            end
            RPE = newQ - Q(x, u);
            Q(x, u) = newQ;
            newElement = [x ; u ; R.hatR(x,u) ; RPE ; abs(RPE) ; argmax(R.hatP(x,u,:)) ; 0];
        end
        sequence = [sequence newElement];
        % we search for predecessors of x
        if (abs(RPE) > R.replayiterthreshold) % high priority
            bufferRPE = searchPredecessor(R, bufferRPE, replayMethod, V, Q, x);
        end
        % we count one more step of VI or PI
        iii = iii + 1;
        if ((iii >= R.window)||(isempty(bufferRPE)))
            letsContinue = false;
        end
    end
end