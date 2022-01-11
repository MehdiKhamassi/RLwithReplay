%%%%%%%%%%%%
%% main.m %%
%%%%%%%%%%%%
% 
% This is the main script for the code to simulate model-based (MB) and
% model-free (MF) reinforcement learning algorithms with replays (i.e.,
% either reactivations of episodic memory buffer during learning phase for
% MF algorithms, or mental simulations of (state,action,new_state,reward)
% quadruplet events with the internal model during inference phase for MB
% algorithms).
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 30 July 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

% %clear all
% %clearvars -except replayMethod M R jjj kkk tabBeta tabPerfBeta
% % for www=[2 5:5:25]
% %     if (www ~= 10)
% %         for kkk=0 %1:5
% %             if (kkk ~= 3)
% %                 for jjj=1:10
% %                     [www kkk jjj]
% %                     main
% %                     close all
% %                     save(['ReversalExperiment2019_model19_window' num2str(www) '_epsilon10-' num2str(kkk) '/ReversalExperiment2019_model19_window' num2str(www) '_epsilon10-' num2str(kkk) '_Expe' num2str(jjj) '.mat'])
% %                     clearvars -except www jjj kkk
% %                 end
% %             end
% %         end
% %     end
% % end
% % www = 10;
% % for kkk=[5 10 20 50 100]
% %     for jjj=1:10
% %         [kkk jjj]
% %         main
% %         close all
% %         save(['ReversalExperiment2019_model19_window10_epsilon10-3_budget' num2str(kkk) '/ReversalExperiment2019_model19_window10_epsilon10-3_budget' num2str(kkk) '_Expe' num2str(jjj) '.mat'])
% %         clearvars -except www jjj kkk
% %     end
% % end
% clc
% [www kkk jjj]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters of the replay experiment %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
replayMethod = 20;
% 0 MF-RL no replay
% 1 basic VI (MB) loop on nS and nA (VI: Value Iteration)
% 2 basic PI (MB) loop on nS and nA (PI: Policy Iteration)
% 3 MF-RL forward (episode window)
% 4 prior MF reset bufferRPE (not used)
% 5 MF-RL prior
% 6 prior MB reset bufferRPE (not used)
% 7 MF-RL shuffl
% 8 MB-RL shuffl
% 9 MF-RL backward (Lin 1992)
% 10 MB-RL trajectory sampling
% 11 MB-RL prioritized sweeping (state-based)
% 12 MB-RL gamma prioritized sweeping (pred: gamma.T(s,u,s').Delta)
% 13 MB-RL bidirectional planning
% 14 Dyna-RL shuffled (Sutton 1990)
% 15 Dyna-RL trajectory
% 16 Dyna-RL bidirectional
% 17 Dyna-RL prioritized sweeping (Moore & Atkeson 1992; Peng & Williams 1993)
% 18 MB-RL prioritized sweeping combined with policy iteration (PI)
% 19 MB-RL prioritized sweeping combined with value iteration (VI)
% 20 MB-RL prioritized sweeping (state,action-based)

% define the Markov Decision Process (MDP) used in the multiple-T-maze task
% of the group of Dave Redish:
M = MDP();
M.totalDuration = 5000; % total duration of the experiment (in timesteps)
M.conditionDuration = 2000; % timesteps after which we change condition
M.logSequenceLength = 2000; % max length of stored replay sequences
M.constraint = 1; % 1 only forward moves, 0 no wall bump, -1 no constraint
M.departureState = 25; % departure state
M.replayPosition = 23; % states where replays are allowed:
% = 1:54; % replays allowed in all states
% = M.stata; % replays allowed in all accessible states (corridors)
% = 25; % replays allowed only at departure state in central arm
% = [6 54]; % replays allowed only at reward sites (6: left; 54: right)
M.P0 = zeros(1,M.nS); % reset distribution of states where replay is allowed
M.P0(M.replayPosition) = 1 / length(M.replayPosition); % allowing replay in some states
M.stochastic = false; % setting reward to be either stochastic or deterministic
M.r = zeros(M.nS, M.nA);
if (M.stochastic)
    M.r(5,2) = 0.75; % on left arm, reward can occur only during transition from state 5 to 6 through action 2 (go south)
    M.r(53,2) = 0.25; % on right arm, from state 53 to 54 with action 2 (go south)
else
    M.r(5,2) = 1; % after a rule shift, this will be set to r(53,2) = 1;
end

% define replay agent
R = replayAgent(M);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRETRAINING to learn the world model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% otherwise the agent does not know the consequence of actions that are not
% allowed during the task (like going backward in the corridor) which
% obviously the real rats have been able to learn during pretraining
for iii=1:10000 % fixed pre-training duration of 10000 timesteps
    if (iii == 1)
        x = drand01(M.P0); % random choice among possible departure states
    else
        x = y; % state encountered after previous action
    end
    acta = possibleMoves(x, 0); % possible actions (except bumping into the walls)
    u = acta(randi(length(acta), 1, 1)); % random choice
    [y, r] = MDPStep(M, x, u); % observing consequence in the environment
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% model-based learning
    % Update the number of visits for the current state-action pair
    R.N(x, u) = R.N(x, u) + 1;
    % Update transition matrix (stochastic)
    R.hatP(x, u, :) = (1 - 1/R.N(x, u)) * R.hatP(x, u, :) + reshape((1:R.nS == y) / R.N(x, u), 1, 1, R.nS);
%     % no reward during pretraining
%     % Update reward function (stochastic)
%     R.hatR(x, u) = (1 - 1/R.N(x, u)) * R.hatR(x, u) + r / R.N(x, u);
end
end_of_pretraining = true

%%%%%%%%%%%%%%%%%%%%%%
%% TASK PERFORMANCE %%
%%%%%%%%%%%%%%%%%%%%%%
[R, logs] = simulateRLwithReplay(M, R, replayMethod);
end_of_experiment = true
% R contains the new state of the agent at the end of the experiment
% logs contains the following:
% logs.memorySweeps stores L or R for each sweep
% logs.memorySide stores L or R for each state replayed
% logs.replaySequence stores the full sequence of replay every even line, and
% info about the sequence (iter replayiter durationReplaySequence) every odd line

%%%%%%%%%%%%%
%% FIGURES %%
%%%%%%%%%%%%%

%% plotting Qmax in each state of the maze at the end of the experiment
Q = R.Q;
plotGradient
% superimposing trajectories (interesting to visualize only if the experiment
% was short; otherwise trajectories are undistinguishable)
hold on
[x1, y1] = stateToCoordinate(M, logs.sequence(1,1)); % starting position
x1 = x1 + rand/5-0.05; % adding a small noise for better visualization
y1 = y1 + rand/5-0.05; % adding a small noise for better visualization
for iii=2:length(logs.sequence(1,:)) % loop over all simulation timesteps
    if (logs.sequence(1,iii) ~= logs.sequence(1,iii-1)) % only during state changes
        [x2, y2] = stateToCoordinate(M, logs.sequence(1,iii)); % arrival position
        x2 = x2 + rand/5-0.05; % adding a small noise for better visualization
        y2 = y2 + rand/5-0.05; % adding a small noise for better visualization
        plot([x1 x2],[y1 y2],'k') % plotting the segment of the trajectory
        x1 = x2; y1 = y2; % starting position <- arrival position
    end
end
xticks([])
yticks([])

%% plotting reward and replay time after each reward followed by ITI
figure

subplot(2,1,1)
imagesc(logs.sequence(5,:))
axis([1 length(logs.sequence(5,:)) 0.5 1.5])
ylabel('task rule')

subplot(2,1,2)
logs.sequence(3,logs.sequence(3,:)<=0) = NaN;
if (size(unique(logs.sequence(5,2:end-1)),1) > 1) % task rule shift
    for iii=2:size(logs.sequence(5,1:end-1),1)
        if (logs.sequence(5,iii) ~= logs.sequence(5,iii-1))
            plot([iii-0.5 iii-0.5],[0 max(logs.sequence(4,:))*1.1],'--','Color', [0.5 0.5 0.5])
            hold on
        end
    end
    plot(logs.sequence(3,:)'*2*max(logs.sequence(4,:))/3,'.k','LineWidth',3)
else % no task rule shift
    plot(logs.sequence(3,:)'*2*max(logs.sequence(4,:))/3,'.k','LineWidth',3)
    hold on
end
plot(logs.sequence(4,:)','Color', [216/255 82/255 24/255],'LineWidth',2)
%axis([1 length(logs.sequence(5,:)) 0 max(logs.sequence(4,:))*1.1])
if (size(unique(logs.sequence(5,:)),1) > 1) % task rule shift
    legend('task rule change','reward','replay duration','location','NorthEast')
else % no task rule shift
    legend('reward','replay duration','location','NorthEast')
end
xlabel('iterations')
ylabel('replay iteration count')

%% plotting learning curve and replay time during ITI
counters = [0 0 0 0 0 0 0]; % each line wil be a trial and the columns represent:
% R/L RWD TRIAL-DURATION REPLAY-DURATION RULE PROP_L-R_SWEEPS PROP_L-R_SIDES
[bbb,iii] = max(logs.sequence(1,:)==23);
counter = iii; %iii = 1; counter = 1;
reward = 0;
ruleChange = [];
cumulSweeps = zeros(1,2);
cumulSides = zeros(1,2);
cumulReplay = 0;
while (iii < length(logs.sequence(1,:)))
    % record if at least one occurence of reward during the trial
    if (logs.sequence(3,iii) == 1)
        reward = 1;
    end
    % record the replay duration during iteration iii
    cumulReplay = cumulReplay + logs.sequence(4,iii);
    
    if (replayMethod >= 3)
        % record if a sweep occurred during the iteration
        if (sum(logs.memorySweeps(:,1)==iii) > 0)
            cumulSweeps(1) = cumulSweeps(1) + logs.memorySweeps(logs.memorySweeps(:,1)==iii,2);
            cumulSweeps(2) = cumulSweeps(2) + logs.memorySweeps(logs.memorySweeps(:,1)==iii,3);
        end
        % record if a sweep occurred during the iteration
        if (sum(logs.memorySide(:,1)==iii) > 0)
            cumulSides(1) = cumulSides(1) + logs.memorySide(logs.memorySide(:,1)==iii,2);
            cumulSides(2) = cumulSides(2) + logs.memorySide(logs.memorySide(:,1)==iii,3);
        end
    end
    % check if end of trial
    if ((logs.sequence(1,iii) == M.departureState)&&(logs.sequence(1,iii) ~= logs.sequence(1,iii-1)))
        % we record possible task rule changes
        if ((size(counters,1) > 1)&&(counters(end,5) ~= logs.sequence(5,iii-1)))
            ruleChange = [ruleChange ; [size(counters,1) iii]]; % we store the trial number
        end
        % we store the characteristics of the trial
        counters = [counters ; [logs.sequence(1,iii-1) reward counter cumulReplay logs.sequence(5,iii-1) 0 0]];
        counter = 0;
        reward = 0;
        cumulReplay = 0;
    end
    if (replayMethod >= 3)
        % check if end of sweeping period (state 23)
        if ((logs.sequence(1,iii) == 23)&&(logs.sequence(1,iii) ~= logs.sequence(1,iii-1)))
            if ((cumulSweeps(1)+cumulSweeps(2)) > 0)
                %[iii logs.sequence(:,iii)' cumulSweeps cumulSweeps(1)/(cumulSweeps(1)+cumulSweeps(2))]
                counters(end,6) = cumulSweeps(1)/(cumulSweeps(1)+cumulSweeps(2));
            else
                counters(end,6) = NaN;
            end
            cumulSweeps = zeros(1,2);
            if ((cumulSides(1)+cumulSides(2)) > 0)
                %[iii logs.sequence(:,iii)' cumulSweeps cumulSweeps(1)/(cumulSweeps(1)+cumulSweeps(2))]
                counters(end,7) = cumulSides(1)/(cumulSides(1)+cumulSides(2));
            else
                counters(end,7) = NaN;
            end
            cumulSides = zeros(1,2);
        end
    end
    % increase counters
    iii = iii + 1;
    counter = counter + 1;
end
counters(1,:) = [];

figure

subplot(3,1,1)
if (size(ruleChange,1) > 0) % task rule shift
    for iii=1:size(ruleChange,1)
        plot([ruleChange(iii,1)-0.5 ruleChange(iii,1)-0.5],[0 max([counters(:,4);counters(:,3)])*1.1],'--','Color', [0.5 0.5 0.5])
        hold on
    end
    plot(counters(:,3:4),'LineWidth',2)
else % no task rule shift
    plot(counters(:,3:4),'LineWidth',2)
    hold on
end
plot(counters(:,2)*2*max([counters(:,4);counters(:,3)])/3,'.k')
axis([1 size(counters(:,5),1) 0 max([counters(:,4);counters(:,3)])*1.1])
if (size(ruleChange,1) > 0) % there is at least one task rule shift
    legend('task rule change','trial duration','cumulated replay duration','reward','Location','southeast')
else % no task rule shift
    legend('trial duration','cumulated replay duration','reward','Location','southeast')
end
ylabel('iteration count')
if (replayMethod >= 3)
    subplot(3,1,2)
    % proportion of sweeps towards left vs right
    %figure,plot(smooth(logs.memorySweeps(:,2)./(logs.memorySweeps(:,2)+logs.memorySweeps(:,3))),'LineWidth',2)
    plot(counters(:,6),'k')
    hold on
    %plot(smooth(counters(:,6),20),'LineWidth',2)
    ylabel('prop L/R sweeps')
    xlabel('trials')
    axis([1 size(counters(:,5),1) 0 1])
    
    subplot(3,1,3)
    % proportion of replays on left vs right side
    plot(counters(:,7),'k')
    hold on
    %plot(smooth(counters(:,7),20),'LineWidth',2)
    ylabel('prop L/R side of replays')
    xlabel('trials')
    axis([1 size(counters(:,5),1) 0 1])
else
    xlabel('trials')
end

% %% figures only for certain replay methods
% if (replayMethod >= 4)
%     %% figure showing the distance to goal for each replayed state within a sequence
%     figure
%     nbcases = 14; %120; %size(logs.replaySequence,1)/2 + 1;
%     change_occurred = 0;
%     goal = 5;
%     average = ones(1,R.window);
%     for iii=1:nbcases
%         subplot(10,12,iii)
%         if ((~isempty(ruleChange))&&(change_occurred == 0)&&(logs.replaySequence(2*(iii-1)+1,1) > ruleChange(1,2)))
%             % rule change
%             imagesc(0)
%             goal = 53;
%             change_occurred = 1;
%         else
%             vector = logs.replaySequence(2*(iii-1)+2,1:logs.replaySequence(2*(iii-1)+1,3));
%             for jjj=1:length(vector)
%                 vector(jjj) = distanceToGoal(goal, vector(jjj));
%             end
%             plot(vector,'k')
%             if (length(vector)<size(average,2))
%                 [iii size(average) size(vector) length(vector)+1] % log
%                 average(:,length(vector)+1:end) = [];
%             end
%             %[size(average) size(vector) length(vector)+1] % log
%             average = [average ; vector(1:size(average,2))];
%         end
%     end
%     average(1,:) = [];
%     
%     %% spatial plots of replays
%     figure
%     goal = 5; pointer = 14; %seq_length = 38;
%     seq_length = logs.replaySequence((pointer-1)*2+1,3);
%     vector = logs.replaySequence((pointer-1)*2+2,1:seq_length);
%     vectaz = vector;
%     for jjj=1:length(vector)
%         vectaz(jjj) = distanceToGoal(goal, vector(jjj));
%     end
%     %[vector ; vectaz] %logs
%     mask = zeros(6,9);mask(2,[2:4 6:8])=-1;mask(3,[2 6:8])=-1;mask(4,[2:3 5:8])=-1;mask(5,[2:3 5:8])=-1;
%     for jjj=2:seq_length/2 %floor(seq_length/3) %(seq_length-2) % 4:9
%         subplot(6,6,jjj-1) % subplot(6,6,24+jjj-1)
%         mask2 = mask;
%         
%         % 1 by 1
%         [x, y] = stateToCoordinate(M, vector(jjj));
%         mask2(y,x) = 1;
% %         % 2 by 2
% %         [x, y] = stateToCoordinate(M, vector(2*(jjj-1)+1));
% %         mask2(y,x) = 1;
% %         [x, y] = stateToCoordinate(M, vector(2*jjj));
% %         mask2(y,x) = 1;
% %         % 3 by 3
% %         [x, y] = stateToCoordinate(M, vector(3*(jjj-1)+1));
% %         mask2(y,x) = 1;
% %         [x, y] = stateToCoordinate(M, vector(3*(jjj-1)+2));
% %         mask2(y,x) = 1;
% %         [x, y] = stateToCoordinate(M, vector(3*jjj));
% %         mask2(y,x) = 1;
%         imagesc(mask2)
%         xticklabels('')
%         yticklabels('')
%         title(['t' num2str(jjj-1)])
%     end
% end

%% figure showing locations of the maze where most replays occurred
nbLines = 1; % one model per line
cLine = 1; % current line
normalizedPlot = true;
family = 2; %'MF' % 2; %'MB' % 3; %'DYNA'
replayLocation = zeros(6,9);
replayLocation1 = zeros(6,9); % earlyLearning
replayLocation2 = zeros(6,9); % lateLearning
replayLocation3 = zeros(6,9); % postShift
replayLocation4 = zeros(6,9); % finalPerf
t1 = floor(M.conditionDuration/2);
t2 = M.conditionDuration;
t3 = M.conditionDuration + floor(M.conditionDuration/2);
t4 = M.totalDuration - floor(M.conditionDuration/2);
nonReplayLocation = ones(6,9);
for iii=1:M.nS % loop over state number
    x = mod(iii,6);
    y = floor((iii-1)/6)+1;
    if (x == 0)
        x = 6;
    end
    logs.sequence(8,:) = 1:length(logs.sequence);
    replayLocation(x,y) = replayLocation(x,y) + sum(logs.sequence(4,logs.sequence(1,:)==iii));
    replayLocation1(x,y) = replayLocation1(x,y) + sum(logs.sequence(4,(logs.sequence(1,:)==iii)&(logs.sequence(8,:)<=t1)));
    replayLocation2(x,y) = replayLocation2(x,y) + sum(logs.sequence(4,(logs.sequence(1,:)==iii)&(logs.sequence(8,:)>t1)&(logs.sequence(8,:)<=t2)));
    replayLocation3(x,y) = replayLocation3(x,y) + sum(logs.sequence(4,(logs.sequence(1,:)==iii)&(logs.sequence(8,:)>t2)&(logs.sequence(8,:)<=t3)));
    replayLocation4(x,y) = replayLocation4(x,y) + sum(logs.sequence(4,(logs.sequence(1,:)==iii)&(logs.sequence(8,:)>t3)&(logs.sequence(8,:)<=t4)));
    % we tag states that are outside the corridors
    if (sum(M.stata==iii) == 0)
        nonReplayLocation(x,y) = -1;
    end
end
if (normalizedPlot)
    theMax = max(max(max(sum(sum(replayLocation1)),sum(sum(replayLocation2))),sum(sum(replayLocation3))),sum(sum(replayLocation4)));
    replayLocation = replayLocation / sum(sum(replayLocation));
    replayLocation1 = replayLocation1 / theMax;
    replayLocation2 = replayLocation2 / theMax;
    replayLocation3 = replayLocation3 / theMax;
    replayLocation4 = replayLocation4 / theMax;
    theMax = max(max(max(max(max(replayLocation1)),max(max(replayLocation2))),max(max(replayLocation3))),max(max(replayLocation4)));
end

% REPLAY LOCATION PER TASK PHASE
if (cLine == 1)
    figure
end
% early learning
subplot(nbLines,4,(cLine-1) * 4 + 1)
h = imagesc(replayLocation1);
switch (cLine)
    case 1
        title('early learning')
        switch(family)
            case 1 % MF
                ylabel('MF-unord')
            case 2 % MB
                ylabel('MB-unord')
            case 3 % DYNA
                ylabel('DYNA-unord')
        end
    case 2
        switch(family)
            case 1 % MF
                ylabel('MF-prior')
            case 2 % MB
                ylabel('MB-prior')
            case 3 % DYNA
                ylabel('DYNA-prior')
        end
    case 3
        switch(family)
            case 1 % MF
                ylabel('MF-forw')
            case 2 % MB
                ylabel('MB-traj')
            case 3 % DYNA
                ylabel('DYNA-traj')
        end
    case 4
        switch(family)
            case 1 % MF
                ylabel('MF-back')
            case 2 % MB
                ylabel('MB-bidir')
            case 3 % DYNA
                ylabel('DYNA-bidir')
        end
end
c = colorbar;
if (normalizedPlot)
    caxis([0 theMax])
end
% plotting the borders of the maze
plotMazeBorders
% late learning
subplot(nbLines,4,(cLine-1) * 4 + 2)
h = imagesc(replayLocation2);
if (cLine == 1)
    title('late learning')
end
c = colorbar;
if (normalizedPlot)
    caxis([0 theMax])
end
% plotting the borders of the maze
plotMazeBorders
% post shift
subplot(nbLines,4,(cLine-1) * 4 + 3)
h = imagesc(replayLocation3);
if (cLine == 1)
    title('post rule change')
end
c = colorbar;
if (normalizedPlot)
    caxis([0 theMax])
end
% plotting the borders of the maze
plotMazeBorders
% final performance
subplot(nbLines,4,(cLine-1) * 4 + 4)
h = imagesc(replayLocation4);
if (cLine == 1)
    title('late experiment')
end
c = colorbar;
if (normalizedPlot)
    caxis([0 theMax])
end
% plotting the borders of the maze
plotMazeBorders
FigHandle = gcf;
if (nbLines == 1)
    set(FigHandle, 'Position', [250, 350, 700, 100]);
else
    set(FigHandle, 'Position', [250, 450, 700, 400]);
end
%print('-bestfit','MBtrajectoryReplayCaze2018_reversal_replayLocation','-dpdf')

% GLOBAL REPLAY LOCATION
if (cLine == 1)
    figure
end
subplot(1,4,cLine)
h = imagesc(replayLocation);
c = colorbar;
switch (cLine)
    case 1
        ylabel('global')
        switch(family)
            case 1 % MF
                title('MF-unord')
            case 2 % MB
                title('MB-unord')
            case 3 % DYNA
                title('DYNA-unord')
        end
    case 2
        switch(family)
            case 1 % MF
                title('MF-prior')
            case 2 % MB
                title('MB-prior')
            case 3 % DYNA
                title('DYNA-prior')
        end
    case 3
        switch(family)
            case 1 % MF
                title('MF-forw')
            case 2 % MB
                title('MB-traj')
            case 3 % DYNA
                title('DYNA-traj')
        end
    case 4
        switch(family)
            case 1 % MF
                title('MF-back')
            case 2 % MB
                title('MB-bidir')
            case 3 % DYNA
                title('DYNA-bidir')
        end
end
%c.Label.String = 'normalized total replay duration in each state (a.u.)';
% plotting the borders of the maze
plotMazeBorders
FigHandle = gcf;
set(FigHandle, 'Position', [250, 350, 700, 100]);
