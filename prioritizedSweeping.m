function [Q, sequence, bufferRPE] = prioritizedSweeping(R, sequence, bufferRPE, replayMethod, V, Q)
% This function performs prioritized sweeping with a
% Markovian model of the environment, and a memory buffer containing all
% surprising events in episodic memory, associated to the amount of
% surprise, that is, reward prediction error
%
% INPUTS:
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
                    pred(1) = [];
                end
            end
        end
        % we reorder bufferRPE depending on priority
        [~, index] = sort(bufferRPE(5,:),'descend');
        bufferRPE = bufferRPE(:,index);
        % we count one more step of VI or PI
        iii = iii + 1;
        if ((iii >= R.window)||(isempty(bufferRPE)))
            letsContinue = false;
        end
    end
end