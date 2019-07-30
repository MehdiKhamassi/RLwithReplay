% plotting the borders of the maze
set(h, 'AlphaData', nonReplayLocation)
hold on
plot([1.5 1.5],[1.5 5.5],'k','LineWidth',2)
plot([1.5 3.5],[5.5 5.5],'k','LineWidth',2)
plot([3.5 3.5],[3.5 5.5],'k','LineWidth',2)
plot([2.5 3.5],[3.5 3.5],'k','LineWidth',2) % if multiple T-maze
plot([2.5 2.5],[2.5 3.5],'k','LineWidth',2) % if multiple T-maze
plot([2.5 3.5],[2.5 2.5],'k','LineWidth',2) % if multiple T-maze
%plot([3.5 3.5],[3.5 2.5],'k','LineWidth',2) % if 8-maze
plot([3.5 4.5],[2.5 2.5],'k','LineWidth',2)
plot([4.5 4.5],[1.5 2.5],'k','LineWidth',2)
plot([1.5 4.5],[1.5 1.5],'k','LineWidth',2)
plot([5.5 5.5],[1.5 3.5],'k','LineWidth',2)
plot([4.5 5.5],[3.5 3.5],'k','LineWidth',2)
plot([4.5 4.5],[3.5 5.5],'k','LineWidth',2)
plot([4.5 8.5],[5.5 5.5],'k','LineWidth',2)
plot([8.5 8.5],[1.5 5.5],'k','LineWidth',2)
plot([5.5 8.5],[1.5 1.5],'k','LineWidth',2)
xticks([])
yticks([])
