function x = distanceToGoal(goal, state)
% This function returns the distance (in number of steps) to one of the two
% reward sites as goal from current state
%
% INPUTS:
%     goal is the state number of one of the reward sites
%     state is the current state
%
% OUTPUTS:
%     x is the number of steps between state and goal
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

    %% x = distance in number of steps to reach
    if (goal == 5)
        switch(state)
            case 1
                x = 4;
            case 2
                x = 3;
            case 3
                x = 2;
            case 4
                x = 1;
            case 5
                x = 0;
            case 6
                x = 1;
            case 7
                x = 5;
            case 12
                x = 2;
            case 13
                x = 6;
            case 15
                x = 8;
            case 18
                x = 3;
            case 19
                x = 7;
            case 21
                x = 7;
            case 22
                x = 6;
            case 23
                x = 5;
            case 24
                x = 4;
            case 25
                x = 8;
            case 26
                x = 9;
            case 27
                x = 8;
            case 30
                x = 5;
            case 31
                x = 9;
            case 36
                x = 6;
            case 37
                x = 10;
            case 42
                x = 7;
            case 43
                x = 11;
            case 48
                x = 8;
            case 49
                x = 12;
            case 50
                x = 13;
            case 51
                x = 12;
            case 52
                x = 11;
            case 53
                x = 10;
            case 54
                x = 9;
            otherwise
                x = -1;
        end
    end % end of if goal == 5
    if (goal == 53)
        switch(state)
            case 1
                x = 12;
            case 2
                x = 13;
            case 3
                x = 12;
            case 4
                x = 11;
            case 5
                x = 10;
            case 6
                x = 9;
            case 7
                x = 11;
            case 12
                x = 8;
            case 13
                x = 10;
            case 15
                x = 10;
            case 18
                x = 7;
            case 19
                x = 9;
            case 21
                x = 9;
            case 22
                x = 8;
            case 23
                x = 7;
            case 24
                x = 6;
            case 25
                x = 8;
            case 26
                x = 9;
            case 27
                x = 10;
            case 30
                x = 5;
            case 31
                x = 7;
            case 36
                x = 4;
            case 37
                x = 6;
            case 42
                x = 3;
            case 43
                x = 5;
            case 48
                x = 2;
            case 49
                x = 4;
            case 50
                x = 3;
            case 51
                x = 2;
            case 52
                x = 1;
            case 53
                x = 0;
            case 54
                x = 1;
            otherwise
                x = -1;
        end
    end % end of if goal == 53
end