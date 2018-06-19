function [x, y] = stateToCoordinate(M, state)
% This function simply translates the state number within the
% multiple-T-maze into (x,y) coordinates (for figure plotting)
%
% INPUTS:
%     M is a structure containing the MDP
%     state is a natural number between 1 and M.nS
%
% OUTPUTS:
%     x x-coordinate
%     y y-coordinate
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr

    %% x coordinate
    if (state <= 6)
        x = 1;
    else if (state <= 12)
        x = 2;
    else if (state <= 18)
        x = 3;
    else if (state <= 24)
        x = 4;
    else if (state <= 30)
        x = 5;
    else if (state <= 36)
        x = 6;
    else if (state <= 42)
        x = 7;
    else if (state <= 48)
        x = 8;
    else
        x = 9;
    end; end; end; end; end; end; end; end

    %% y coordinate
    y = mod(state,6); %7 - mod(state,6); % if y-axis inverted
    if (y == 0)
        y = 6;
    end
end