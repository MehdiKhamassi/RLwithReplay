% This scripts plots a figure showing the gradient of max Q-values learned
% by a reinforcement learning model in each state of the multiple-T-maze
% task of A. David Redish and colleagues
%
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

figure,imagesc([max(Q(1:6,:)')' max(Q(7:12,:)')' max(Q(13:18,:)')' max(Q(19:24,:)')' max(Q(25:30,:)')' max(Q(31:36,:)')' max(Q(37:42,:)')' max(Q(43:48,:)')' max(Q(49:54,:)')'])
c = colorbar;
c.Label.String = 'maximum Q-value in each state (a.u.)';
