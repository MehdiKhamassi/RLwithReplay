%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIGURE AVERAGING 10 EXPERIMENTS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% This is a script generating article figures for 10 simulation experiments
% with a particular replayMethod in one of the model-free or model-based
% reinforcement learning models tested in the multiple-T-maze task of A.
% David Redish and colleagues.
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 30 July 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

modelName = 'model20_window54_epsilon10-3_budgetInf'; %'model19_MBpriorVI_newPriorSamplRPEpred_TrajSamplUntilVisiting1stateOfBufferRPE_window10replayThreshold001_strategy2_beta3'; % 'model19_window25'; %'model11_MBprior_RPEpred_beta3'; % 'model10_MBtraj_beta3'; % 
preFixDirectoryName = 'Massi2022_'; %'~/Documents/_owncloud/travaux/ISIR/replay/newExpeMay2019/ReversalExperiment2019_'; % '~/Documents/_owncloud/travaux/ISIR/replay/Caze2017_allExperiments/Caze2017_Experiments_'; % 'newExpeDec2018/ReversalExperiment2018_'; % 
preFixFileName = 'Massi2022_'; % 'Caze2017_'; % 'ReversalExperiment2018_'; % 
curveColor = 'k'; %[224/255 224/255 224/255]; % %one color per model, if one wants to superimpose the curves of different models for comparison
barPos = 11:14; % one position per model: either 1:4 or 6:9 or 11:14 or 16:19
nbExpe = 10; % number of simulation experiments
durationPreShift = 100; % (number of trials before task rule change)
durationPostShift = 100; % (number of trials after task rule change)
dataAlreadySaved = false; % if true, data previously saved with this function will be loaded, otherwise they will be generated and then saved in a file

if (~dataAlreadySaved)
    % buffers to be initialized once per algo
    duration = [];
    replayDuration = [];
    rwdrate = [];
    propsweep = [];
    propside = [];
    forwardsweep = []; % fwd bwd img rnd TOTAL (3 states)
    forwardsweep5 = []; % fwd bwd img rnd TOTAL (5 states)
    % we also count sweeps when position of the animal is in central arm
    forwardsweep_centralArm = []; % fwd bwd img rnd TOTAL (3 states)
    forwardsweep5_centralArm = []; % fwd bwd img rnd TOTAL (5 states)
    % we also count sweeps when position of the animal is at reward location
    forwardsweep_rwdLocation = []; % fwd bwd img rnd TOTAL (3 states)
    forwardsweep5_rwdLocation = []; % fwd bwd img rnd TOTAL (5 states)
    
    glogs.sequence = []; % global logs concatenating logs of each of the 10 experiments

    for iii=1:nbExpe % 10 experiments per algo
        % for each experiment, load data
        load([preFixDirectoryName modelName '/' preFixFileName modelName '_Expe' num2str(iii) '.mat'])
        
        glogs.sequence = [glogs.sequence logs.sequence];
        % rename compteurs for old experiments of Caze Khamassi et al 2018:
        %counters = compteurs;
        % then process and store the experiment's data in the buffers
        %boubou = counters(:,3)+max(0,counters(:,4)-100); % old 2018 for VI
        % 2019: for traj sampl, we exclude the first M.window iterations
        % that are automatically done after each action to check Q-value
        % converge and decide whether to do more replay cycles:
        switch (replayMethod)
            case 19
                boubou = counters(:,3)+max(0,counters(:,4)-counters(:,3)*2*R.window); % at each trial: window=10 traj sampl + window=10 for prior sweep
            otherwise
                boubou = counters(:,3)+max(0,counters(:,4)-counters(:,3)*R.window);
        end
        % replay methods where the minimum is always 100 replay cycles
        %boubou = counters(:,3)+counters(:,4); % new 2019 for PI methods
        TC = argmax(counters(:,5)==53); % For reversal experiments
%         % for extinction experiments: we search for the first extinction trial
%         nbNoR = 0;
%         for jjj=1:size(counters(:,3))
%             if (counters(jjj,2) == 1) % rewarded trial
%                 nbNoR = 0;
%             else % unrewarded trial
%                 if (nbNoR == 0)
%                     nbNoR = 1;
%                 else
%                     nbNoR = nbNoR + 1;
%                     if (nbNoR >= 20) % 20 consecutive unrewarded trials
%                         TC = jjj-nbNoR+1; % we identified the first extinction trial
%                         break;
%                     end
%                 end
%             end
%         end
        % extracting pre- and post-shift data
        duration = [duration ; [boubou([1:durationPreShift TC:TC+durationPostShift-1])']];
        replayDuration = [replayDuration ; [counters([1:durationPreShift TC:TC+durationPostShift-1],4)']];
        rwdrate = [rwdrate ; [counters([1:durationPreShift TC:TC+durationPostShift-1],2)']];
        propsweep = [propsweep ; [counters([1:durationPreShift TC:TC+durationPostShift-1],6)']];
        propside = [propside ; [counters([1:durationPreShift TC:TC+durationPostShift-1],6)']];

        nbcases = size(logs.replaySequence,1)/2; % + 1;
        propfwd3 = zeros(1,5); % fwd bwd img rnd TOTAL (3 states)
        propfwd5 = zeros(1,5); % fwd bwd img rnd TOTAL (5 states)
        propfwd3_centralArm = zeros(1,5); % fwd bwd img rnd TOTAL (3 states)
        propfwd5_centralArm = zeros(1,5); % fwd bwd img rnd TOTAL (5 states)
        propfwd3_rwdLocation = zeros(1,5); % fwd bwd img rnd TOTAL (3 states)
        propfwd5_rwdLocation = zeros(1,5); % fwd bwd img rnd TOTAL (5 states)
        
        % loop over total number of replay events (one event including a vector of replayed (state,action) pairs)
        for kkk=1:nbcases
            durationReplaySequence = min(2000,logs.replaySequence(2*(kkk-1)+1,3));
            vecteur = logs.replaySequence(2*(kkk-1)+2,1:durationReplaySequence);
            current_location = logs.sequence(1,logs.replaySequence(2*(kkk-1)+1,1));
            alreadycounted3 = zeros(size(vecteur)); % here we tag elements already categorized as replays
            alreadycounted5 = zeros(size(vecteur)); % same thing for sequences of 5 elements
            compteur3 = 1; % we save all 3 states
            compteur5 = 1; % we save all 5 states
            sensdusweep = 0; % 0 nothing 1 forward 2 backward 3 imaginary
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % we first search for img replays
            % considering each (state,action) pair of the replay event
            for jjj=1:length(vecteur)
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % we first search for img sweeps on 3 consecutive states
                    
                % if the element has already been categorized as part of a
                % replay of 3 elements, we skip it
                if (alreadycounted3(jjj))
                    compteur3 = 1; % reinit counter and skip element
                else
                    if (compteur3 >= 3)
                        if (((vecteur(jjj-2)==19)&&(vecteur(jjj-1)==25)&&(vecteur(jjj)==31))||((vecteur(jjj-2)==31)&&(vecteur(jjj-1)==25)&&(vecteur(jjj)==19))||((vecteur(jjj-2)==18)&&(vecteur(jjj-1)==24)&&(vecteur(jjj)==30))||((vecteur(jjj-2)==30)&&(vecteur(jjj-1)==24)&&(vecteur(jjj)==18)))
                            sensdusweep = 3;
                            propfwd3(3) = propfwd3(3) + 1; % img replay
                            switch (current_location)
                                case {15,21,22,23,24,25,26,27} % central arm
                                    propfwd3_centralArm(3) = propfwd3_centralArm(3) + 1; % img replay
                                case {3,4,5,6,51,52,53,54} % reward location
                                    propfwd3_rwdLocation(3) = propfwd3_rwdLocation(3) + 1; % img replay
                            end
                            alreadycounted3(jjj-2:jjj) = 1;
                            compteur3 = 1; % reinit counter
                        end
                    else
                        compteur3 = compteur3 + 1;
                    end
                end
                
                % if the element has already been categorized as part of a
                % replay of 5 elements, we skip it
                if (alreadycounted5(jjj))
                    compteur5 = 1; % reinit counter and skip element
                else
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % we now search for img sweeps on 5 consecutive states
                    if (compteur5 >= 5)
                        % we first check if imaginary forward sweep
                        if (((vecteur(jjj-4)==13)&&(vecteur(jjj-3)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-1)==31)&&(vecteur(jjj)==37))||((vecteur(jjj-4)==37)&&(vecteur(jjj-3)==31)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-1)==19)&&(vecteur(jjj)==13))||((vecteur(jjj-4)==12)&&(vecteur(jjj-3)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-1)==30)&&(vecteur(jjj)==36))||((vecteur(jjj-4)==36)&&(vecteur(jjj-3)==30)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-1)==18)&&(vecteur(jjj)==12)))
                            sensdusweep = 3;
                            propfwd5(3) = propfwd5(3) + 1; % img replay
                            switch (current_location)
                                case {15,21,22,23,24,25,26,27} % central arm
                                    propfwd5_centralArm(3) = propfwd5_centralArm(3) + 1; % img replay
                                case {3,4,5,6,51,52,53,54} % reward location
                                    propfwd5_rwdLocation(3) = propfwd5_rwdLocation(3) + 1; % img replay
                            end
                            alreadycounted5(jjj-4:jjj) = 1;
                            compteur5 = 1; % reinit counter
                        end
                    else
                        compteur5 = compteur5 + 1;
                    end
                end
                
            end % end of the loop on all elements of the replay event
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % we then search for other types of replays (fwd, bwd, other)
            % considering each (state,action) pair of the replay event
            compteur3 = 1;
            compteur5 = 1;
            for jjj=1:length(vecteur)
                
                % if the element has already been categorized as part of a
                % replay of 3 elements, we skip it
                if (alreadycounted3(jjj))
                    compteur3 = 1; % reinit counter and skip element
                else
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % we first search for replays on 3 consecutive states
                    if (compteur3 >= 3)
                        % we check if forward replay
                        if ((isForward(vecteur(jjj-2),vecteur(jjj-1))==1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==1))
                            % we first check if imaginary sweep
                            if (((vecteur(jjj-2)==19)&&(vecteur(jjj-1)==25)&&(vecteur(jjj)==31))||((vecteur(jjj-2)==31)&&(vecteur(jjj-1)==25)&&(vecteur(jjj)==19))||((vecteur(jjj-2)==18)&&(vecteur(jjj-1)==24)&&(vecteur(jjj)==30))||((vecteur(jjj-2)==30)&&(vecteur(jjj-1)==24)&&(vecteur(jjj)==18)))
                                sequenceRejouee3 = [vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
                                sensdusweep = 3;
                                propfwd3(3) = propfwd3(3) + 1; % img replay
                                switch (current_location)
                                    case {15,21,22,23,24,25,26,27} % central arm
                                        propfwd3_centralArm(3) = propfwd3_centralArm(3) + 1; % img replay
                                    case {3,4,5,6,51,52,53,54} % reward location
                                        propfwd3_rwdLocation(3) = propfwd3_rwdLocation(3) + 1; % img replay
                                end
                                alreadycounted3(jjj-2:jjj) = 1;
                                compteur3 = 1; % reinit counter
                            else
                                sensdusweep = 1;
                                propfwd3(1) = propfwd3(1) + 1; % fwd replay
                                switch (current_location)
                                    case {15,21,22,23,24,25,26,27} % central arm
                                        propfwd3_centralArm(1) = propfwd3_centralArm(1) + 1; % fwd replay
                                    case {3,4,5,6,51,52,53,54} % reward location
                                        propfwd3_rwdLocation(1) = propfwd3_rwdLocation(1) + 1; % fwd replay
                                end
                                alreadycounted3(jjj-2:jjj) = 1;
                                compteur3 = 1; % reinit counter
                            end
                        else
                            % we check if backward replay
                            if ((isForward(vecteur(jjj-2),vecteur(jjj-1))==-1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==-1))
                                % we first check if imaginary sweep
                                if (((vecteur(jjj-2)==19)&&(vecteur(jjj-1)==25)&&(vecteur(jjj)==31))||((vecteur(jjj-2)==31)&&(vecteur(jjj-1)==25)&&(vecteur(jjj)==19))||((vecteur(jjj-2)==18)&&(vecteur(jjj-1)==24)&&(vecteur(jjj)==30))||((vecteur(jjj-2)==30)&&(vecteur(jjj-1)==24)&&(vecteur(jjj)==18)))
                                    sequenceRejouee3 = [vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
                                    sensdusweep = 3;
                                    propfwd3(3) = propfwd3(3) + 1; % img replay
                                    switch (current_location)
                                        case {15,21,22,23,24,25,26,27} % central arm
                                            propfwd3_centralArm(3) = propfwd3_centralArm(3) + 1; % img replay
                                        case {3,4,5,6,51,52,53,54} % reward location
                                            propfwd3_rwdLocation(3) = propfwd3_rwdLocation(3) + 1; % img replay
                                    end
                                    alreadycounted3(jjj-2:jjj) = 1;
                                    compteur3 = 1; % reinit counter
                                else
                                    sensdusweep = -1;
                                    propfwd3(2) = propfwd3(2) + 1; % bwd replay
                                    switch (current_location)
                                        case {15,21,22,23,24,25,26,27} % central arm
                                            propfwd3_centralArm(2) = propfwd3_centralArm(2) + 1; % bwd replay
                                        case {3,4,5,6,51,52,53,54} % reward location
                                            propfwd3_rwdLocation(2) = propfwd3_rwdLocation(2) + 1; % bwd replay
                                    end
                                    alreadycounted3(jjj-2:jjj) = 1;
                                    compteur3 = 1; % reinit counter
                                end
                            else
%                                 if ((isForward(vecteur(jjj-2),vecteur(jjj-1))~=isForward(vecteur(jjj-1),vecteur(jjj)))&&((((vecteur(jjj-2)==19)&&(vecteur(jjj)==31))||((vecteur(jjj-2)==31)&&(vecteur(jjj)==19))||((vecteur(jjj-2)==18)&&(vecteur(jjj)==30))||((vecteur(jjj-2)==30)&&(vecteur(jjj)==18)))))
%                                     sequenceRejouee3 = [vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
%                                     sensdusweep = 3;
%                                     propfwd3(3) = propfwd3(3) + 1; % img replay
%                                     alreadycounted3(jjj-2:jjj) = 1;
%                                     compteur3 = 1; % reinit counter
%                                 else
%                                     % nothing for the moment
%                                     % let's look one step ahead to see if the
%                                     % replay is shifted
%                                 end
                            end
                        end
                    else
                        compteur3 = compteur3 + 1;
                    end
                end % end of if alreadycounted3(jjj)
                
                % if the element has already been categorized as part of a
                % replay of 5 elements, we skip it
                if (alreadycounted5(jjj))
                    compteur5 = 1; % reinit counter and skip element
                else
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % we now search for img sweeps on 5 consecutive states
                    if (compteur5 >= 5)
                        % we check if forward replay
                        if ((isForward(vecteur(jjj-4),vecteur(jjj-3))==1)&&(isForward(vecteur(jjj-3),vecteur(jjj-2))==1)&&(isForward(vecteur(jjj-2),vecteur(jjj-1))==1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==1))
                            % we first check if imaginary forward sweep
                            if (((vecteur(jjj-4)==13)&&(vecteur(jjj-3)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-1)==31)&&(vecteur(jjj)==37))||((vecteur(jjj-4)==37)&&(vecteur(jjj-1)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-3)==31)&&(vecteur(jjj)==13))||((vecteur(jjj-4)==12)&&(vecteur(jjj-3)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-1)==30)&&(vecteur(jjj)==36))||((vecteur(jjj-4)==36)&&(vecteur(jjj-1)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-3)==30)&&(vecteur(jjj)==12)))
                                sequenceRejouee5 = [vecteur(jjj-4) vecteur(jjj-3) vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
                                sensdusweep = 3;
                                propfwd5(3) = propfwd5(3) + 1; % img replay
                                switch (current_location)
                                    case {15,21,22,23,24,25,26,27} % central arm
                                        propfwd5_centralArm(3) = propfwd5_centralArm(3) + 1; % img replay
                                    case {3,4,5,6,51,52,53,54} % reward location
                                        propfwd5_rwdLocation(3) = propfwd5_rwdLocation(3) + 1; % img replay
                                end
                                alreadycounted5(jjj-4:jjj) = 1;
                                compteur5 = 1; % reinit counter
                            else % simple forward sweep
                                sensdusweep = 1;
                                propfwd5(1) = propfwd5(1) + 1; % fwd replay
                                switch (current_location)
                                    case {15,21,22,23,24,25,26,27} % central arm
                                        propfwd5_centralArm(1) = propfwd5_centralArm(1) + 1; % fwd replay
                                    case {3,4,5,6,51,52,53,54} % reward location
                                        propfwd5_rwdLocation(1) = propfwd5_rwdLocation(1) + 1; % fwd replay
                                end
                                alreadycounted5(jjj-4:jjj) = 1;
                                compteur5 = 1; % reinit counter
                            end
                        else
                            % we check if backward replay
                            if ((isForward(vecteur(jjj-4),vecteur(jjj-3))==-1)&&(isForward(vecteur(jjj-3),vecteur(jjj-2))==-1)&&(isForward(vecteur(jjj-2),vecteur(jjj-1))==-1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==-1))
                                % we first check if imaginary backward sweep
                                if (((vecteur(jjj-4)==13)&&(vecteur(jjj-3)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-1)==31)&&(vecteur(jjj)==37))||((vecteur(jjj-4)==37)&&(vecteur(jjj-1)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-3)==31)&&(vecteur(jjj)==13))||((vecteur(jjj-4)==12)&&(vecteur(jjj-3)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-1)==30)&&(vecteur(jjj)==36))||((vecteur(jjj-4)==36)&&(vecteur(jjj-1)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-3)==30)&&(vecteur(jjj)==12)))
                                    sequenceRejouee5 = [vecteur(jjj-4) vecteur(jjj-3) vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
                                    sensdusweep = 3;
                                    propfwd5(3) = propfwd5(3) + 1; % img replay
                                    switch (current_location)
                                        case {15,21,22,23,24,25,26,27} % central arm
                                            propfwd5_centralArm(3) = propfwd5_centralArm(3) + 1; % img replay
                                        case {3,4,5,6,51,52,53,54} % reward location
                                            propfwd5_rwdLocation(3) = propfwd5_rwdLocation(3) + 1; % img replay
                                    end
                                    alreadycounted5(jjj-4:jjj) = 1;
                                    compteur5 = 1; % reinit counter
                                else % simple backward sweep
                                    sensdusweep = -1;
                                    propfwd5(2) = propfwd5(2) + 1; % bwd replay
                                    switch (current_location)
                                        case {15,21,22,23,24,25,26,27} % central arm
                                            propfwd5_centralArm(2) = propfwd5_centralArm(2) + 1; % bwd replay
                                        case {3,4,5,6,51,52,53,54} % reward location
                                            propfwd5_rwdLocation(2) = propfwd5_rwdLocation(2) + 1; % bwd replay
                                    end
                                    alreadycounted5(jjj-4:jjj) = 1;
                                    compteur5 = 1; % reinit counter
                                end
                            else
                                if (((isForward(vecteur(jjj-4),vecteur(jjj-3))==-1)&&(isForward(vecteur(jjj-3),vecteur(jjj-2))==-1)&&(isForward(vecteur(jjj-2),vecteur(jjj-1))==1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==1))||((isForward(vecteur(jjj-4),vecteur(jjj-3))==1)&&(isForward(vecteur(jjj-3),vecteur(jjj-2))==1)&&(isForward(vecteur(jjj-2),vecteur(jjj-1))==-1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==-1)))
                                    if (((vecteur(jjj-4)==13)&&(vecteur(jjj-3)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-1)==31)&&(vecteur(jjj)==37))||((vecteur(jjj-4)==37)&&(vecteur(jjj-1)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-3)==31)&&(vecteur(jjj)==13))||((vecteur(jjj-4)==12)&&(vecteur(jjj-3)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-1)==30)&&(vecteur(jjj)==36))||((vecteur(jjj-4)==36)&&(vecteur(jjj-1)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-3)==30)&&(vecteur(jjj)==12)))
                                        sequenceRejouee5 = [vecteur(jjj-4) vecteur(jjj-3) vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
                                        sensdusweep = 3;
                                        propfwd5(3) = propfwd5(3) + 1; % img replay
                                        switch (current_location)
                                            case {15,21,22,23,24,25,26,27} % central arm
                                                propfwd5_centralArm(3) = propfwd5_centralArm(3) + 1; % img replay
                                            case {3,4,5,6,51,52,53,54} % reward location
                                                propfwd5_rwdLocation(3) = propfwd5_rwdLocation(3) + 1; % img replay
                                        end
                                        alreadycounted5(jjj-4:jjj) = 1;
                                        compteur5 = 1; % reinit counter
                                    else
                                        % nothing for the moment
                                        % let's look one step ahead to see if the
                                        % replay is shifted
                                    end
                                else
                                    % nothing for the moment
                                    % let's look one step ahead to see if the
                                    % replay is shifted
                                end
                            end
                        end
                    else
                        compteur5 = compteur5 + 1;
                    end
                end % end of if alreadycounted5(jjj)
                
            end % end of the loop on all elements of the replay event
            % NEW 2018:
            % we look at all uncategorized replayed elements and count them
            % as 1/3 or 1/5 of "other replays" (because considered replays here
            % have a length of 3 or 5 elements)
            propfwd3(4) = propfwd3(4) + sum(alreadycounted3==0) / 3;
            propfwd5(4) = propfwd5(4) + sum(alreadycounted5==0) / 5;
            switch (current_location)
                case {15,21,22,23,24,25,26,27} % central arm
                    propfwd3_centralArm(4) = propfwd3_centralArm(4) + sum(alreadycounted3==0) / 3;
                    propfwd5_centralArm(4) = propfwd5_centralArm(4) + sum(alreadycounted5==0) / 5;
                case {3,4,5,6,51,52,53,54} % reward location
                    propfwd3_rwdLocation(4) = propfwd3_rwdLocation(4) + sum(alreadycounted3==0) / 3;
                    propfwd5_rwdLocation(4) = propfwd5_rwdLocation(4) + sum(alreadycounted5==0) / 5;
            end
            checkresult = 1;
            if (propfwd5(3) > propfwd3(3)) % impossible!
                checkresult = 0
            end
        end % end of the loop on all replay events
        propfwd3(5) = sum(propfwd3(1:4));
        propfwd5(5) = sum(propfwd5(1:4));
        propfwd3_centralArm(5) = sum(propfwd3_centralArm(1:4));
        propfwd5_centralArm(5) = sum(propfwd5_centralArm(1:4));
        propfwd3_rwdLocation(5) = sum(propfwd3_rwdLocation(1:4));
        propfwd5_rwdLocation(5) = sum(propfwd5_rwdLocation(1:4));
        checkresult = 1;
        forwardsweep = [forwardsweep ; propfwd3];
        forwardsweep5 = [forwardsweep5 ; propfwd5];
        forwardsweep_centralArm = [forwardsweep_centralArm ; propfwd3_centralArm];
        forwardsweep5_centralArm = [forwardsweep5_centralArm ; propfwd5_centralArm];
        forwardsweep_rwdLocation = [forwardsweep_rwdLocation ; propfwd3_rwdLocation];
        forwardsweep5_rwdLocation = [forwardsweep5_rwdLocation ; propfwd5_rwdLocation];
    end % end of loop on the 10 experiments for this algo

    % saving the data of the 10 experiments for the considered algo
    save([preFixDirectoryName modelName '/' preFixFileName modelName '_dataFrom' num2str(nbExpe) 'experiments.mat'], 'propsweep', 'propside', 'duration', 'replayDuration', 'forwardsweep', 'forwardsweep5','rwdrate')
else % else of if (~dataAlreadySaved)
    load([preFixDirectoryName modelName '/' preFixFileName modelName '_dataFrom' num2str(nbExpe) 'experiments.mat'])
end % end of if (~dataAlreadySaved)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
% plot learning curves
subplot(2,1,1)
plot([10000 10100],[50 50],'Color',[57/255 83/255 164/255],'LineWidth',3) % blue
%plot([10000 10100],[50 50],'k','LineWidth',3) % black
hold on
%plot([10000 10100],[50 50],'Color',[102/255 102/255 102/255],'LineWidth',3) % blue
%plot([10000 10100],[50 50],'Color',[235/255 32/255 38/255],'LineWidth',3) % red
plot([10000 10100],[50 50],'Color',[253/255 122/255 10/255],'LineWidth',3) % orange
plot([10000 10100],[50 50],'Color',[106/255 188/255 69/255],'LineWidth',3) % green
%plot([10000 10100],[50 50],'Color',[134/255 197/255 252/255],'LineWidth',3) % cyan
plot([10000 10100],[50 50],'k','LineWidth',3)
%plot([10000 10100],[50 50],'Color',[51/255 51/255 51/255],'LineWidth',3) % blue
% plot([10000 10100],[50 50],'Color',[153/255 153/255 153/255],'LineWidth',3) % blue
% plot([10000 10100],[50 50],'Color',[170/255 170/255 170/255],'LineWidth',3) % blue
% plot([10000 10100],[50 50],'Color',[204/255 204/255 204/255],'LineWidth',3) % blue
% plot([10000 10100],[50 50],'Color',[224/255 224/255 224/255],'LineWidth',3) % blue
plot([durationPreShift durationPreShift],[0 15],'--','Color', [0.5 0.5 0.5])
errorfill(1:durationPreShift+durationPostShift,mean(log(duration)),std(log(duration)),curveColor)
%plot(1:durationPreShift+durationPostShift,mean(log(duration)),'Color',curveColor,'LineWidth',3)
axis([0 durationPreShift+durationPostShift 0 15]) % for reversal experiments
%axis([0 durationPreShift+durationPostShift 2 8]) % for extinction experiments
ylabel('ln(# model iterations)','FontSize',16)
% BELOW, CHOOSE THE APPROPRIATE TITLE DEPENDING ON WHAT'S BEING PLOTTED IN
% THE FIGURE, AND COMMENT THE UNAPPROPRIATE TITLES:
%title('MF-RL. green: abs(RPE)-prioritized replays')
%title('MF-RL. black: no replay; blue: unordered replays; red: backward replays')
%title('MB-RL. blue: unordered replays; green: prioritized sweeping')
%legend('MF-RL no replays','MF-RL backward replays')
%legend('MB-RL unordered inference','MB-RL prioritized sweeping','MB-RL trajectory sampling')
%legend('MB-RL prioritized sweeping')
%legend('MF-RL unordered','MF-RL backward','MF-RL forward','MF-RL prior')
%legend('MB-RL shuffled','DYNA shuffled')
%legend('MB-RL PTSPI','MB-RL PTSVI') % Prioritized Trajectory Sampling with PI or VI
%legend('MB-RL bidirectional planning') % Prioritized Trajectory Sampling with PI or VI
%legend('MB-RL trajectory sampling','MB-RL prioritized sweeping','MB-RL bidirectional planning') % Prioritized Trajectory Sampling with PI or VI
%legend('MF-RL no replay','MF-RL backward','MF-RL shuffle','MB-RL prior S','MB-RL prior SA')
legend('MF-RL no replay','MF-RL backward','MF-RL shuffle','MB-RL prior sweep')
%legend('budget=2','budget=5','budget=10','budget=15','budget=20','budget=25')
%legend('budget=25','budget=10','budget=2')
%legend('epsilon=0.1','epsilon=0.01','epsilon=0.001','epsilon=0.0001','epsilon=0.00001')
%legend('epsilon=0.1','epsilon=0.01','epsilon=0.001')
%legend('infinite budget','budget=5','budget=10','budget=20','budget=50','budget=100')
alpha 0.5

% % plot L/R sweeps
% subplot(3,1,2)
% plot([durationPreShift durationPreShift],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
% hold on
% propsweep2 = propsweep;
% propsweep2(isnan(propsweep2)) = 0.5;
% errorfill(1:durationPreShift+durationPostShift,mean(propsweep2),std(propsweep2),curveColor)
% axis([0 durationPreShift+durationPostShift -0.5 1.5])
% ylabel('prop L/R sweeps')
% alpha 0.5

% plot reward rate
subplot(2,1,2)
plot([durationPreShift durationPreShift],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
hold on
errorfill(1:durationPreShift+durationPostShift,mean(rwdrate),std(rwdrate),curveColor)
%plot(1:durationPreShift+durationPostShift,mean(rwdrate),'Color',curveColor,'LineWidth',3)
axis([0 durationPreShift+durationPostShift -0.5 1.5])
ylabel('reward rate','FontSize',16)
xlabel('trial','FontSize',16)
alpha 0.5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
% plot L/R sides
subplot(3,1,1)
plot([durationPreShift-0.5 durationPreShift-0.5],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
hold on
propside2 = propside;
propswide2(isnan(propside2)) = 0.5;
errorfill(1:durationPreShift+durationPostShift,mean(propside2),std(propside2),curveColor)
axis([0 durationPreShift+durationPostShift -0.5 1.5])
ylabel('prop L/R side of replays','FontSize',16)

% plot L/R sweeps
subplot(3,1,2)
plot([durationPreShift-0.5 durationPreShift-0.5],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
hold on
propsweep2 = propsweep;
propsweep2(isnan(propsweep2)) = 0.5;
errorfill(1:durationPreShift+durationPostShift,mean(propsweep2),std(propsweep2),curveColor)
axis([0 durationPreShift+durationPostShift -0.5 1.5])
ylabel('prop L/R sweeps','FontSize',16)
xlabel('trial','FontSize',16)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
% plot percentage forward/backward/rand sweeps on 3 consecutive states
subplot(3,1,1)
forwardsweep2 = forwardsweep(:,1:4);
for iii=1:nbExpe
    forwardsweep2(iii,1:4) = 100 * forwardsweep2(iii,1:4) / forwardsweep(iii,5);
end
bar(barPos(1),mean(forwardsweep2(:,1)), 'c')
hold on
bar(barPos(2),mean(forwardsweep2(:,2)), 'r')
bar(barPos(3),mean(forwardsweep2(:,3)), 'g')
bar(barPos(4),mean(forwardsweep2(:,4)), 'b')
if strcmp('MF',modelName(1:2)) % the figure represents model-free models
    legend('forward replays','backward replays','imaginary replays','other replays')
else % the figure represents model-based models
    legend('forward inference','backward inference','imaginary inference','other inference')
end
errorbar(barPos(1), mean(forwardsweep2(:,1)), std(forwardsweep2(:,1)), '+k')
errorbar(barPos(2), mean(forwardsweep2(:,2)), std(forwardsweep2(:,2)), '+k')
errorbar(barPos(3), mean(forwardsweep2(:,3)), std(forwardsweep2(:,3)), '+k')
errorbar(barPos(4), mean(forwardsweep2(:,4)), std(forwardsweep2(:,4)), '+k')
plot([5 5],[-10 110],'--','Color', [0.5 0.5 0.5])
plot([10 10],[-10 110],'--','Color', [0.5 0.5 0.5])
plot([15 15],[-10 110],'--','Color', [0.5 0.5 0.5])
xticks([])
yticks([0 25 50 75 100])
axis([0 20 -10 110])
ylabel('3s sweep prop')
% BELOW, CHOOSE THE APPROPRIATE TITLE DEPENDING ON WHAT4S BEING PLOTTED IN
% THE FIGURE, AND COMMENT THE UNAPPROPRIATE TITLES:
%title('MB-RL. blue: unordered replays; green: prioritized sweeping')
%title('MF-RL. blue: unordered; red: backward; cyan: forward; green: prioritized.')
%title('  MF-RL-unordered   MF-RL-backward      MF-RL-forward         MF-RL-prior    ')
%title('    MF-RL-prior            MF-RL-forward     MF-RL-backward  MF-RL-unordered    ')
%title('      MB-RL-prior   MB-RL-trajectory   MB-RL-bidirectional    MB-RL-unordered    ')
%title('    DYNA-RL-prior DYNA-RL-bidirectional DYNA-RL-trajectory  DYNA-RL-unordered    ')
%title('      MB-RL-prior   MB-RL-trajectory   MB-RL-PTSVI    MB-RL-PTSPI    ')
title('      MB-RL-trajectory   MB-RL-prioritized   MB-RL-bidirectional    ')

% plot percentage forward/backward/rand sweeps on 5 consecutive states
subplot(3,1,2)
forwardsweep2 = forwardsweep5(:,1:4);
for iii=1:nbExpe
    forwardsweep2(iii,1:4) = 100 * forwardsweep2(iii,1:4) / forwardsweep5(iii,5);
end
bar(barPos(1),mean(forwardsweep2(:,1)), 'c')
hold on
bar(barPos(2),mean(forwardsweep2(:,2)), 'r')
bar(barPos(3),mean(forwardsweep2(:,3)), 'g')
bar(barPos(4),mean(forwardsweep2(:,4)), 'b')
errorbar(barPos(1), mean(forwardsweep2(:,1)), std(forwardsweep2(:,1)), '+k')
errorbar(barPos(2), mean(forwardsweep2(:,2)), std(forwardsweep2(:,2)), '+k')
errorbar(barPos(3), mean(forwardsweep2(:,3)), std(forwardsweep2(:,3)), '+k')
errorbar(barPos(4), mean(forwardsweep2(:,4)), std(forwardsweep2(:,4)), '+k')
plot([5 5],[-10 110],'--','Color', [0.5 0.5 0.5])
plot([10 10],[-10 110],'--','Color', [0.5 0.5 0.5])
plot([15 15],[-10 110],'--','Color', [0.5 0.5 0.5])
xticks([])
yticks([0 25 50 75 100])
axis([0 20 -10 110])
ylabel('5s sweep prop')

%% WE PLOT SWEEP PROPORTIONS IN DIFFERENT POSITIONS:
%% 1 GLOBAL (ALL POSITIONS)
%% 2 WHEN POSITION OF THE ANIMAL==CENTRAL ARM
%% 3 WHEN POSITION OF THE ANIMAL==REWARD LOCATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
% plot percentage forward/backward/rand sweeps on 3 consecutive states
subplot(3,1,1)
forwardsweep2 = forwardsweep(:,1:4);
forwardsweep_centralArm(forwardsweep_centralArm(:,5)==0,:) = []; % avoid division by 0
forwardsweep_rwdLocation(forwardsweep_rwdLocation(:,5)==0,:) = []; % avoid division by 0
forwardsweep2_ca = forwardsweep_centralArm(:,1:4);
forwardsweep2_rl = forwardsweep_rwdLocation(:,1:4);
for iii=1:nbExpe
    forwardsweep2(iii,1:4) = 100 * forwardsweep2(iii,1:4) / forwardsweep(iii,5);
    if (size(forwardsweep2_ca,1) >= iii)
        forwardsweep2_ca(iii,1:4) = 100 * forwardsweep2_ca(iii,1:4) / forwardsweep_centralArm(iii,5);
    end
    if (size(forwardsweep2_rl,1) >= iii)
        forwardsweep2_rl(iii,1:4) = 100 * forwardsweep2_rl(iii,1:4) / forwardsweep_rwdLocation(iii,5);
    end
end
global_prop = [sum([mean(forwardsweep2(:,1)) mean(forwardsweep2(:,2)) mean(forwardsweep2(:,3)) mean(forwardsweep2(:,4))]) mean(forwardsweep2(:,1)) mean(forwardsweep2(:,2)) mean(forwardsweep2(:,3)) mean(forwardsweep2(:,4))]
central_prop = [sum([mean(forwardsweep2_ca(:,1)) mean(forwardsweep2_ca(:,2)) mean(forwardsweep2_ca(:,3)) mean(forwardsweep2_ca(:,4))]) mean(forwardsweep2_ca(:,1)) mean(forwardsweep2_ca(:,2)) mean(forwardsweep2_ca(:,3)) mean(forwardsweep2_ca(:,4))]
reward_prop = [sum([mean(forwardsweep2_rl(:,1)) mean(forwardsweep2_rl(:,2)) mean(forwardsweep2_rl(:,3)) mean(forwardsweep2_rl(:,4))]) mean(forwardsweep2_rl(:,1)) mean(forwardsweep2_rl(:,2)) mean(forwardsweep2_rl(:,3)) mean(forwardsweep2_rl(:,4))]
bar(1,mean(forwardsweep2(:,1)), 'c')
hold on
bar(2,mean(forwardsweep2(:,2)), 'r')
bar(3,mean(forwardsweep2(:,3)), 'g')
bar(4,mean(forwardsweep2(:,4)), 'b')
% if strcmp('MF',modelName(1:2)) % the figure represents model-free models
%     legend('forward replays','backward replays','imaginary replays','other replays')
% else % the figure represents model-based models
%     legend('forward inference','backward inference','imaginary inference','other inference')
% end
errorbar(1, mean(forwardsweep2(:,1)), std(forwardsweep2(:,1)), '+k')
errorbar(2, mean(forwardsweep2(:,2)), std(forwardsweep2(:,2)), '+k')
errorbar(3, mean(forwardsweep2(:,3)), std(forwardsweep2(:,3)), '+k')
errorbar(4, mean(forwardsweep2(:,4)), std(forwardsweep2(:,4)), '+k')
% central arm
bar(6,mean(forwardsweep2_ca(:,1)), 'c')
bar(7,mean(forwardsweep2_ca(:,2)), 'r')
bar(8,mean(forwardsweep2_ca(:,3)), 'g')
bar(9,mean(forwardsweep2_ca(:,4)), 'b')
errorbar(6, mean(forwardsweep2_ca(:,1)), std(forwardsweep2_ca(:,1)), '+k')
errorbar(7, mean(forwardsweep2_ca(:,2)), std(forwardsweep2_ca(:,2)), '+k')
errorbar(8, mean(forwardsweep2_ca(:,3)), std(forwardsweep2_ca(:,3)), '+k')
errorbar(9, mean(forwardsweep2_ca(:,4)), std(forwardsweep2_ca(:,4)), '+k')
% reward locations
bar(11,mean(forwardsweep2_rl(:,1)), 'c')
bar(12,mean(forwardsweep2_rl(:,2)), 'r')
bar(13,mean(forwardsweep2_rl(:,3)), 'g')
bar(14,mean(forwardsweep2_rl(:,4)), 'b')
errorbar(11, mean(forwardsweep2_rl(:,1)), std(forwardsweep2_rl(:,1)), '+k')
errorbar(12, mean(forwardsweep2_rl(:,2)), std(forwardsweep2_rl(:,2)), '+k')
errorbar(13, mean(forwardsweep2_rl(:,3)), std(forwardsweep2_rl(:,3)), '+k')
errorbar(14, mean(forwardsweep2_rl(:,4)), std(forwardsweep2_rl(:,4)), '+k')
plot([5 5],[-10 110],'--','Color', [0.5 0.5 0.5])
plot([10 10],[-10 110],'--','Color', [0.5 0.5 0.5])
xticks([])
yticks([0 25 50 75 100])
axis([0 15 -10 110])
ylabel('3s sweep prop')
title('      ALL LOCATIONS    CENTRAL ARM   REWARD LOCATIONS    ')
myfig = gcf;

% TWO-WAY ANOVA sweep type x location
donnees3 = [forwardsweep2_ca(1:min(size(forwardsweep2_ca,1),size(forwardsweep2_rl,1)),:);forwardsweep2_rl(1:min(size(forwardsweep2_ca,1),size(forwardsweep2_rl,1)),:)];

gcf = myfig;
% plot percentage forward/backward/rand sweeps on 5 consecutive states
subplot(3,1,2)
forwardsweep2 = forwardsweep5(:,1:4);
forwardsweep5_centralArm(forwardsweep5_centralArm(:,5)==0,:) = []; % avoid division by 0
forwardsweep5_rwdLocation(forwardsweep5_rwdLocation(:,5)==0,:) = []; % avoid division by 0
forwardsweep2_ca = forwardsweep5_centralArm(:,1:4);
forwardsweep2_rl = forwardsweep5_rwdLocation(:,1:4);
for iii=1:nbExpe
    forwardsweep2(iii,1:4) = 100 * forwardsweep2(iii,1:4) / forwardsweep5(iii,5);
    if (size(forwardsweep2_ca,1) >= iii)
        forwardsweep2_ca(iii,1:4) = 100 * forwardsweep2_ca(iii,1:4) / forwardsweep5_centralArm(iii,5);
    end
    if (size(forwardsweep2_rl,1) >= iii)
        forwardsweep2_rl(iii,1:4) = 100 * forwardsweep2_rl(iii,1:4) / forwardsweep5_rwdLocation(iii,5);
    end
end
global_prop = [sum([mean(forwardsweep2(:,1)) mean(forwardsweep2(:,2)) mean(forwardsweep2(:,3)) mean(forwardsweep2(:,4))]) mean(forwardsweep2(:,1)) mean(forwardsweep2(:,2)) mean(forwardsweep2(:,3)) mean(forwardsweep2(:,4))]
central_prop = [sum([mean(forwardsweep2_ca(:,1)) mean(forwardsweep2_ca(:,2)) mean(forwardsweep2_ca(:,3)) mean(forwardsweep2_ca(:,4))]) mean(forwardsweep2_ca(:,1)) mean(forwardsweep2_ca(:,2)) mean(forwardsweep2_ca(:,3)) mean(forwardsweep2_ca(:,4))]
reward_prop = [sum([mean(forwardsweep2_rl(:,1)) mean(forwardsweep2_rl(:,2)) mean(forwardsweep2_rl(:,3)) mean(forwardsweep2_rl(:,4))]) mean(forwardsweep2_rl(:,1)) mean(forwardsweep2_rl(:,2)) mean(forwardsweep2_rl(:,3)) mean(forwardsweep2_rl(:,4))]
bar(1,mean(forwardsweep2(:,1)), 'c')
hold on
bar(2,mean(forwardsweep2(:,2)), 'r')
bar(3,mean(forwardsweep2(:,3)), 'g')
bar(4,mean(forwardsweep2(:,4)), 'b')
errorbar(1, mean(forwardsweep2(:,1)), std(forwardsweep2(:,1)), '+k')
errorbar(2, mean(forwardsweep2(:,2)), std(forwardsweep2(:,2)), '+k')
errorbar(3, mean(forwardsweep2(:,3)), std(forwardsweep2(:,3)), '+k')
errorbar(4, mean(forwardsweep2(:,4)), std(forwardsweep2(:,4)), '+k')
% central arm
bar(6,mean(forwardsweep2_ca(:,1)), 'c')
bar(7,mean(forwardsweep2_ca(:,2)), 'r')
bar(8,mean(forwardsweep2_ca(:,3)), 'g')
bar(9,mean(forwardsweep2_ca(:,4)), 'b')
errorbar(6, mean(forwardsweep2_ca(:,1)), std(forwardsweep2_ca(:,1)), '+k')
errorbar(7, mean(forwardsweep2_ca(:,2)), std(forwardsweep2_ca(:,2)), '+k')
errorbar(8, mean(forwardsweep2_ca(:,3)), std(forwardsweep2_ca(:,3)), '+k')
errorbar(9, mean(forwardsweep2_ca(:,4)), std(forwardsweep2_ca(:,4)), '+k')
% reward locations
bar(11,mean(forwardsweep2_rl(:,1)), 'c')
bar(12,mean(forwardsweep2_rl(:,2)), 'r')
bar(13,mean(forwardsweep2_rl(:,3)), 'g')
bar(14,mean(forwardsweep2_rl(:,4)), 'b')
errorbar(11, mean(forwardsweep2_rl(:,1)), std(forwardsweep2_rl(:,1)), '+k')
errorbar(12, mean(forwardsweep2_rl(:,2)), std(forwardsweep2_rl(:,2)), '+k')
errorbar(13, mean(forwardsweep2_rl(:,3)), std(forwardsweep2_rl(:,3)), '+k')
errorbar(14, mean(forwardsweep2_rl(:,4)), std(forwardsweep2_rl(:,4)), '+k')
plot([5 5],[-10 110],'--','Color', [0.5 0.5 0.5])
plot([10 10],[-10 110],'--','Color', [0.5 0.5 0.5])
xticks([])
yticks([0 25 50 75 100])
axis([0 15 -10 110])
ylabel('5s sweep prop')

% TWO-WAY ANOVA sweep type x location
donnees5 = [forwardsweep2_ca(1:min(size(forwardsweep2_ca,1),size(forwardsweep2_rl,1)),:);forwardsweep2_rl(1:min(size(forwardsweep2_ca,1),size(forwardsweep2_rl,1)),:)];
[p,tbl,stats] = anova2(donnees3,size(donnees3,1)/2)
[p,tbl,stats] = anova2(donnees5,size(donnees5,1)/2)
clear gcf;

% print('-bestfit',[preFixDirectoryName modelName '/' preFixFileName modelName '_Fig5_perf10experiments'],'-dpdf')

