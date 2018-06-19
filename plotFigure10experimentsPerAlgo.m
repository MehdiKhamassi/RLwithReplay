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
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

modelName = 'MFbackwardReplay';
preFixDirectoryName = 'Caze2017_allExperiments/Caze2017_Experiments_';
preFixFileName = 'Caze2017_';
curveColor = 'b'; % one color per model, if one wants to superimpose the curves of different models for comparison
barPos = 1:4; % one position per model: either 1:4 or 6:9 or 11:14 or 16:19
nbExpe = 10; % number of simulation experiments
durationPreShift = 100; % (number of trials before task rule change)
durationPostShift = 100; % (number of trials after task rule change)
dataAlreadySaved = false; % if true, data previously saved with this function will be loaded, otherwise they will be generated and then saved in a file

if (~dataAlreadySaved)
    % buffers to be initialized once per algo
    duration = [];
    rwdrate = [];
    propsweep = [];
    propside = [];
    forwardsweep = []; % fwd bwd img rnd TOTAL (3 states)
    forwardsweep5 = []; % fwd bwd img rnd TOTAL (5 states)

    for iii=1:nbExpe % 10 experiments per algo
        % for each experiment, load data
        load([preFixDirectoryName modelName '/' preFixFileName modelName '_Expe' num2str(iii) '.mat'])

        % then process and store the experiment's data in the buffers
        boubou = compteurs(:,3)+max(0,compteurs(:,4)-100);
        TC = argmax(compteurs(:,5)==53);
        duration = [duration ; [boubou([1:durationPreShift TC:TC+durationPostShift-1])']];
        rwdrate = [rwdrate ; [compteurs([1:durationPreShift TC:TC+durationPostShift-1],2)']];
        propsweep = [propsweep ; [compteurs([1:durationPreShift TC:TC+durationPostShift-1],6)']];
        propside = [propside ; [compteurs([1:durationPreShift TC:TC+durationPostShift-1],6)']];

        nbcases = size(logs.replaySequence,1)/2; % + 1;
        propfwd3 = zeros(1,5); % fwd bwd img rnd TOTAL (3 states)
        propfwd5 = zeros(1,5); % fwd bwd img rnd TOTAL (5 states)
        
        % loop over total number of replay events (one event including a vector of replayed (state,action) pairs)
        for kkk=1:nbcases
            durationReplaySequence = min(2000,logs.replaySequence(2*(kkk-1)+1,3));
            vecteur = logs.replaySequence(2*(kkk-1)+2,1:durationReplaySequence);
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
                                alreadycounted3(jjj-2:jjj) = 1;
                                compteur3 = 1; % reinit counter
                            else
                                sensdusweep = 1;
                                propfwd3(1) = propfwd3(1) + 1; % fwd replay
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
                                    alreadycounted3(jjj-2:jjj) = 1;
                                    compteur3 = 1; % reinit counter
                                else
                                    sensdusweep = -1;
                                    propfwd3(2) = propfwd3(2) + 1; % bwd replay
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
                                alreadycounted5(jjj-4:jjj) = 1;
                                compteur5 = 1; % reinit counter
                            else % simple forward sweep
                                sensdusweep = 1;
                                propfwd5(1) = propfwd5(1) + 1; % fwd replay
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
                                    alreadycounted5(jjj-4:jjj) = 1;
                                    compteur5 = 1; % reinit counter
                                else % simple backward sweep
                                    sensdusweep = -1;
                                    propfwd5(2) = propfwd5(2) + 1; % bwd replay
                                    alreadycounted5(jjj-4:jjj) = 1;
                                    compteur5 = 1; % reinit counter
                                end
                            else
                                if (((isForward(vecteur(jjj-4),vecteur(jjj-3))==-1)&&(isForward(vecteur(jjj-3),vecteur(jjj-2))==-1)&&(isForward(vecteur(jjj-2),vecteur(jjj-1))==1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==1))||((isForward(vecteur(jjj-4),vecteur(jjj-3))==1)&&(isForward(vecteur(jjj-3),vecteur(jjj-2))==1)&&(isForward(vecteur(jjj-2),vecteur(jjj-1))==-1)&&(isForward(vecteur(jjj-1),vecteur(jjj))==-1)))
                                    if (((vecteur(jjj-4)==13)&&(vecteur(jjj-3)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-1)==31)&&(vecteur(jjj)==37))||((vecteur(jjj-4)==37)&&(vecteur(jjj-1)==19)&&(vecteur(jjj-2)==25)&&(vecteur(jjj-3)==31)&&(vecteur(jjj)==13))||((vecteur(jjj-4)==12)&&(vecteur(jjj-3)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-1)==30)&&(vecteur(jjj)==36))||((vecteur(jjj-4)==36)&&(vecteur(jjj-1)==18)&&(vecteur(jjj-2)==24)&&(vecteur(jjj-3)==30)&&(vecteur(jjj)==12)))
                                        sequenceRejouee5 = [vecteur(jjj-4) vecteur(jjj-3) vecteur(jjj-2) vecteur(jjj-1) vecteur(jjj)]
                                        sensdusweep = 3;
                                        propfwd5(3) = propfwd5(3) + 1; % img replay
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
            checkresult = 1;
            if (propfwd5(3) > propfwd3(3)) % impossible!
                checkresult = 0
            end
        end % end of the loop on all replay events
        propfwd3(5) = sum(propfwd3(1:4));
        propfwd5(5) = sum(propfwd5(1:4));
        checkresult = 1;
        forwardsweep = [forwardsweep ; propfwd3];
        forwardsweep5 = [forwardsweep5 ; propfwd5];
    end % end of loop on the 10 experiments for this algo

    % saving the data of the 10 experiments for the considered algo
    save([preFixDirectoryName modelName '/' preFixFileName modelName '_dataFrom' num2str(nbExpe) 'experiments.mat'], 'propsweep', 'propside', 'duration', 'forwardsweep', 'forwardsweep5','rwdrate')
else % else of if (~dataAlreadySaved)
    load([preFixDirectoryName modelName '/' preFixFileName modelName '_dataFrom' num2str(nbExpe) 'experiments.mat'])
end % end of if (~dataAlreadySaved)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
% plot learning curves
subplot(3,1,1)
plot([10000 10100],[50 50],'k','LineWidth',3)
hold on
plot([10000 10100],[50 50],'Color',[57/255 83/255 164/255],'LineWidth',3) % blue
plot([10000 10100],[50 50],'Color',[235/255 32/255 38/255],'LineWidth',3) % red
plot([10000 10100],[50 50],'Color',[106/255 188/255 69/255],'LineWidth',3) % green
plot([durationPreShift-0.5 durationPreShift-0.5],[0 15],'--','Color', [0.5 0.5 0.5])
errorfill(1:durationPreShift+durationPostShift,mean(log(duration)),std(log(duration)),curveColor)
axis([0 durationPreShift+durationPostShift 0 15])
ylabel('ln(# replay steps)')
% BELOW, CHOOSE THE APPROPRIATE TITLE DEPENDING ON WHAT4S BEING PLOTTED IN
% THE FIGURE, AND COMMENT THE UNAPPROPRIATE TITLES:
%title('MF-RL. green: abs(RPE)-prioritized replays')
%title('MF-RL. black: no replay; blue: shuffled replays; red: backward replays')
%title('MB-RL. blue: shuffled replays; green: prioritized sweeping')
legend('MF-RL no replays','MF-RL backward replays')
%legend('MB-RL shuffled inference','MB-RL prioritized sweeping','MB-RL trajectory sampling')
%legend('MB-RL prioritized sweeping','DYNA-RL prioritized sweeping')
%legend('MF-RL shuffled','MF-RL backward','MF-RL forward','MF-RL prior')
alpha 0.5

% plot L/R sweeps
subplot(3,1,2)
plot([durationPreShift-0.5 durationPreShift-0.5],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
hold on
propsweep2 = propsweep;
propsweep2(isnan(propsweep2)) = 0.5;
errorfill(1:durationPreShift+durationPostShift,mean(propsweep2),std(propsweep2),curveColor)
axis([0 durationPreShift+durationPostShift -0.5 1.5])
ylabel('prop L/R sweeps')
alpha 0.5

% plot reward rate
subplot(3,1,3)
plot([durationPreShift-0.5 durationPreShift-0.5],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
hold on
errorfill(1:durationPreShift+durationPostShift,mean(rwdrate),std(rwdrate),curveColor)
axis([0 durationPreShift+durationPostShift -0.5 1.5])
ylabel('reward rate')
xlabel('trial')
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
ylabel('prop L/R side of replays')

% plot L/R sweeps
subplot(3,1,2)
plot([durationPreShift-0.5 durationPreShift-0.5],[-0.5 1.5],'--','Color', [0.5 0.5 0.5])
hold on
propsweep2 = propsweep;
propsweep2(isnan(propsweep2)) = 0.5;
errorfill(1:durationPreShift+durationPostShift,mean(propsweep2),std(propsweep2),curveColor)
axis([0 durationPreShift+durationPostShift -0.5 1.5])
ylabel('prop L/R sweeps')
xlabel('trial')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
% plot percentage forward/backward/rand sweeps on 3 consecutive states
subplot(3,1,1)
forwardsweep2 = forwardsweep(:,1:4);
for iii=1:10
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
%title('MB-RL. blue: shuffled replays; green: prioritized sweeping')
%title('MF-RL. blue: shuffled; red: backward; cyan: forward; green: prioritized.')
title('  MF-RL-shuffled   MF-RL-backward      MF-RL-forward         MF-RL-prior    ')
%title('      MB-RL-prior   MB-RL-trajectory   MB-RL-bidirectional    MB-RL-shuffled    ')
%title('    DYNA-RL-prior DYNA-RL-bidirectional DYNA-RL-trajectory  DYNA-RL-shuffled    ')

% plot percentage forward/backward/rand sweeps on 3 consecutive states
subplot(3,1,2)
forwardsweep2 = forwardsweep5(:,1:4);
for iii=1:10
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

% print('-bestfit',[preFixDirectoryName modelName '/' preFixFileName modelName '_Fig5_perf10experiments'],'-dpdf')

