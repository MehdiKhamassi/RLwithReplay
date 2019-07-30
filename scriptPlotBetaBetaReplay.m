%% THIS SCRIPT LOADS DATA AND MEASURES WHAT PERCENTAGE OF VARIANCE TO PERFORMANCE AND
%% NUMBER OF REPLAY EACH NEW SIMULATION EXPERIMENT ADDS.
%
%     created 24 May 2019
%     by Mehdi Khamassi
%     last modified 24 May 2019
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

nbExpe = 5;
nbOrder = 20;
difference = zeros(nbOrder, nbExpe-1, 2);
bigTabPerfBeta = zeros(nbExpe, 10, 10, 2);
for ooo=1:nbOrder
    expeOrder = randperm(nbExpe); %[5 4 3 2 1]; % order in which we take experiments for the mean
    load(['model19_tabPerBetaBetaReplay' num2str(expeOrder(1)) '.mat'])
    tabMoyOLD = tabPerfBeta;
    tabMoy = zeros(size(tabPerfBeta));
    bigTabPerfBeta(1, :, :, :) = tabPerfBeta;
    for iii=2:size(expeOrder,2)
        load(['model19_tabPerBetaBetaReplay' num2str(expeOrder(iii)) '.mat'])
        bigTabPerfBeta(iii, :, :, :) = tabPerfBeta;
        for jjj=1:10
            for kkk=1:10
                tabMoy(jjj, kkk, 1) = mean(bigTabPerfBeta(1:iii, jjj, kkk, 1));
                tabMoy(jjj, kkk, 2) = mean(bigTabPerfBeta(1:iii, jjj, kkk, 2));
            end
        end
        difference(ooo, iii-1, :) = [sum(sum(abs(tabMoy(:, :, 1) - tabMoyOLD(:, :, 1)))) * 100 / sum(sum(abs(tabMoyOLD(:, :, 1)))) sum(sum(abs(tabMoy(:, :, 2) - tabMoyOLD(:, :, 2)))) * 100 / sum(sum(abs(tabMoyOLD(:, :, 2))))];
        tabMoyOLD = tabMoy;
    end
end

% load(['model19_tabPerBetaBetaReplay' num2str(expeOrder(1)) '.mat'])
% tabPerfBeta5 = tabPerfBeta;
% tabMoyOLD = tabPerfBeta5;
% load model19_tabPerBetaBetaReplay4.mat
% tabPerfBeta4 = tabPerfBeta;
% tabMoy = zeros(size(tabPerfBeta));
% for jjj=1:10
%     for kkk=1:10
%         tabMoy(jjj,kkk,1) = mean([tabPerfBeta4(jjj,kkk,1) tabPerfBeta5(jjj,kkk,1)]);
%         tabMoy(jjj,kkk,2) = mean([tabPerfBeta4(jjj,kkk,2) tabPerfBeta5(jjj,kkk,2)]);
%     end
% end
% difference = [sum(sum(abs(tabMoy(:,:,1)-tabMoyOLD(:,:,1))))*100/sum(sum(abs(tabMoyOLD(:,:,1)))) sum(sum(abs(tabMoy(:,:,2)-tabMoyOLD(:,:,2))))*100/sum(sum(abs(tabMoyOLD(:,:,2))))];
% tabMoyOLD = tabMoy;
% load model19_tabPerBetaBetaReplay3.mat
% tabPerfBeta3 = tabPerfBeta;
% for jjj=1:10
%     for kkk=1:10
%         tabMoy(jjj,kkk,1) = mean([tabPerfBeta3(jjj,kkk,1) tabPerfBeta4(jjj,kkk,1) tabPerfBeta5(jjj,kkk,1)]);
%         tabMoy(jjj,kkk,2) = mean([tabPerfBeta3(jjj,kkk,2) tabPerfBeta4(jjj,kkk,2) tabPerfBeta5(jjj,kkk,2)]);
%     end
% end
% difference = [difference ; [sum(sum(abs(tabMoy(:,:,1)-tabMoyOLD(:,:,1))))*100/sum(sum(abs(tabMoyOLD(:,:,1)))) sum(sum(abs(tabMoy(:,:,2)-tabMoyOLD(:,:,2))))*100/sum(sum(abs(tabMoyOLD(:,:,2))))]];
% tabMoyOLD = tabMoy;
% load model19_tabPerBetaBetaReplay2.mat
% tabPerfBeta2 = tabPerfBeta;
% for jjj=1:10
%     for kkk=1:10
%         tabMoy(jjj,kkk,1) = mean([tabPerfBeta2(jjj,kkk,1) tabPerfBeta3(jjj,kkk,1) tabPerfBeta4(jjj,kkk,1) tabPerfBeta5(jjj,kkk,1)]);
%         tabMoy(jjj,kkk,2) = mean([tabPerfBeta2(jjj,kkk,2) tabPerfBeta3(jjj,kkk,2) tabPerfBeta4(jjj,kkk,2) tabPerfBeta5(jjj,kkk,2)]);
%     end
% end
% difference = [difference ; [sum(sum(abs(tabMoy(:,:,1)-tabMoyOLD(:,:,1))))*100/sum(sum(abs(tabMoyOLD(:,:,1)))) sum(sum(abs(tabMoy(:,:,2)-tabMoyOLD(:,:,2))))*100/sum(sum(abs(tabMoyOLD(:,:,2))))]];
% tabMoyOLD = tabMoy;
% load model19_tabPerBetaBetaReplay.mat
% for jjj=1:10
%     for kkk=1:10
%         tabMoy(jjj,kkk,1) = mean([tabPerfBeta(jjj,kkk,1) tabPerfBeta2(jjj,kkk,1) tabPerfBeta3(jjj,kkk,1) tabPerfBeta4(jjj,kkk,1) tabPerfBeta5(jjj,kkk,1)]);
%         tabMoy(jjj,kkk,2) = mean([tabPerfBeta(jjj,kkk,2) tabPerfBeta2(jjj,kkk,2) tabPerfBeta3(jjj,kkk,2) tabPerfBeta4(jjj,kkk,2) tabPerfBeta5(jjj,kkk,2)]);
%     end
% end
% difference = [difference ; [sum(sum(abs(tabMoy(:,:,1)-tabMoyOLD(:,:,1))))*100/sum(sum(abs(tabMoyOLD(:,:,1)))) sum(sum(abs(tabMoy(:,:,2)-tabMoyOLD(:,:,2))))*100/sum(sum(abs(tabMoyOLD(:,:,2))))]];

%% FIGURE BETA BETA_REPLAY
tabPerfBeta = tabMoy;
syms beta beta_R
figure
subplot(2,2,1)
imagesc(reshape(tabPerfBeta(:,:,1),10,10))
set(gca,'YDir','normal','XTick',[1:2 4:8 10],'XTickLabels',tabBeta([1:2 4:8 10]),'YTick',1:length(tabBeta),'YTickLabels',tabBeta)
hold on
plot([0 0],[15 15],'-k','LineWidth',3)
colorbar
ylabel(texlabel(beta_R))
xlabel(texlabel(beta))
title('cumulated reward')
subplot(2,2,2)
imagesc(reshape(tabPerfBeta(:,:,2),10,10))
set(gca,'YDir','normal','XTick',[1:2 4:8 10],'XTickLabels',tabBeta([1:2 4:8 10]),'YTick',1:length(tabBeta),'YTickLabels',tabBeta)
colorbar
ylabel(texlabel(beta_R))
xlabel(texlabel(beta))
title('cumulated nb replay iterations')
subplot(2,2,3:4)
errorbar(2:nbExpe,mean(difference(:,:,1)),std(difference(:,:,1)),'k','LineWidth',2,'MarkerSize',2)
hold on
errorbar(2:nbExpe,mean(difference(:,:,2)),std(difference(:,:,2)),'b','LineWidth',2,'MarkerSize',2)
legend('cumulated reward','cumulated nb replay iterations')
axis([1.5 nbExpe+0.5 0 7])
set(gca,'XTick',2:nbExpe,'XTickLabels',2:nbExpe)
xlabel('nb expe per parameter-set')
ylabel('percentage change in the mean')

%% FINDING THE BEST PARAMETER-SET
% with Chebyshev aggregation function (Wierzbicki, 1986)
maxX = max(max(tabPerfBeta(:,:,1)));
minX = min(min(tabPerfBeta(:,:,1)));
tabPerfBeta(:,:,1) = (tabPerfBeta(:,:,1) - minX) / (maxX - minX); % normalization
maxY = max(max(tabPerfBeta(:,:,2)));
minY = min(min(tabPerfBeta(:,:,2)));
tabPerfBeta(:,:,2) = (tabPerfBeta(:,:,2) - minY) / (maxY - minY); % normalization
chebyshev = tabPerfBeta(:,:,1) - tabPerfBeta(:,:,2);
