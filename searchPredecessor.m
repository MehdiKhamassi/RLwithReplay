function bufferRPE = searchPredecessor(R, bufferRPE, replayMethod, V, Q, x)
% This function is related to MB/DYNA prioritized sweeping. It searches for
% predecessors of the current state x, computes their reward prediction
% error RPEpred, and adds them the to the memory buffer bufferRPE
% containing all surprising events in episodic memory, associated to the
% amount of surprise, that is, reward prediction error
%
% INPUTS:
%     R is the replay agent
%     V is the currently estimated value function
%     Q is the currently estimated (state,action) value function
%     bufferRPE contains the list of elements to be replayed, ordered by descending abs(RPE)
%     x is the current state
%     sequence contains the sampled replay trajectory
%
% OUTPUTS:
% 
%     created 19 Dec 2022
%     by Mehdi Khamassi
%     last modified 19 Dec 2022
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 
    
    for aaa = 1:R.nA
        pred = R.hatP(:,aaa,x)>(1/R.nS);
        indexes = (1:R.nS);
        pred = indexes(pred);
        while (~isempty(pred))
            % compute RPE for each pred
            switch (replayMethod)
                case {2,18} % PI methods
                    Qpred = R.hatR(pred(1), aaa) + R.gamma * sum(reshape(R.hatP(pred(1), aaa, :), R.nS, 1)' * V);
                    RPEpred = Qpred - Q(pred(1),aaa);
                case {6,11,12,19} % VI methods
                    Qmax = max(Q, [], 2);
                    Qpred = R.hatR(pred(1), aaa) + R.gamma * sum(reshape(R.hatP(pred(1), aaa, :), R.nS, 1) .* Qmax);
                    RPEpred = Qpred - Q(pred(1),aaa);
                otherwise % MF or Dyna methods
                    Qmax = max(Q, [], 2);
                    RPEpred = R.hatR(pred(1), aaa) + R.gamma * Qmax(x) - Q(pred(1), aaa);
            end
            % add (pred,RPEpred) to bufferRPE
            switch (replayMethod)
                case 20 % (state,action)-based prioritized sweeping
                    if (sum(bufferRPE(1,:)==pred(1)&bufferRPE(2,:)==aaa)==0) % predecessor not already in bufferRPE
                        if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > R.replayiterthreshold) % high priority
                            bufferRPE = [bufferRPE [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0]];
                        end
                    else % predecessor already in bufferRPE
                        if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > bufferRPE(5,bufferRPE(1,:)==pred(1))) % higher priority than previously stored in buffer for this predecessor
                            bufferRPE(:,bufferRPE(1,:)==pred(1)&bufferRPE(2,:)==aaa) = [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0];
                        end
                    end
                otherwise % state-based prioritized sweeping
                    if (sum(bufferRPE(1,:)==pred(1))==0) % predecessor not already in bufferRPE
                        if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > R.replayiterthreshold) % high priority
                            if (replayMethod == 12)
                                bufferRPE = [bufferRPE [pred(1) ; aaa ; 0 ; R.gamma * R.hatP(pred(1),aaa,x) * RPEpred ; R.gamma * R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0]];
                            else
                                bufferRPE = [bufferRPE [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0]];
                            end
                        end
                    else
                        if ((R.hatP(pred(1),aaa,x) * abs(RPEpred)) > bufferRPE(5,bufferRPE(1,:)==pred(1))) % higher priority than previously stored in buffer for this predecessor
                            if (replayMethod == 12)
                                bufferRPE(:,bufferRPE(1,:)==pred(1)) = [pred(1) ; aaa ; 0 ; R.gamma * R.hatP(pred(1),aaa,x) * RPEpred ; R.gamma * R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0];
                            else
                                bufferRPE(:,bufferRPE(1,:)==pred(1)) = [pred(1) ; aaa ; 0 ; R.hatP(pred(1),aaa,x) * RPEpred ; R.hatP(pred(1),aaa,x) * abs(RPEpred) ; x ; 0];
                            end
                        end
                    end
            end
            pred(1) = [];
        end
    end
    % we reorder bufferRPE depending on priority
    [~, index] = sort(bufferRPE(5,:),'descend');
    bufferRPE = bufferRPE(:,index);
end