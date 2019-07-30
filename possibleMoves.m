function x = possibleMoves(state, constraint)
% This function returns the set of possible moves (i.e. actions) from the
% current state in the multiple-T-maze task of A. David Redish and
% colleagues
%
% INPUTS:
%     state is an identification number (between 1 and 54)
%
% OUTPUTS:
%     x is a vector containing all possible actions
%     constraint = 1 if the agent is forced to move forward and prevented
%     to ever go back or to bump into walls; constraint = 0 if the only
%     forbidden thing is to bump into the walls.
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

    %% x = possible actions
    if (constraint == 1)
        switch(state)
            case 1
                x = 2;
            case 2
                x = 2;
            case 3
                x = 2;
            case 4
                x = 2;
            case 5
                x = 2;
            case 6
                x = 3;
            case 7
                x = 4;
            case 12
                x = 3;
            case 13
                x = 4;
            case 15
                x = 3;
            case 18
                x = 3;
            case 19
                x = 4;
            case 21
                x = [3 4]; % if double T-maze
                %x = 3; % if 8-maze
            case 22
                x = 1;
            case 23
                x = 1;
            case 24
                x = 1;
            case 25
                x = [3 4];
            case 26
                x = 1;
            case 27
                x = 1;
            case 30
                x = 4;
            case 31
                x = 3;
            case 36
                x = 4;
            case 37
                x = 3;
            case 42
                x = 4;
            case 43
                x = 3;
            case 48
                x = 4;
            case 49
                x = 2;
            case 50
                x = 2;
            case 51
                x = 2;
            case 52
                x = 2;
            case 53
                x = 2;
            case 54
                x = 4;
            otherwise
                x = [1 2 3 4];
        end
    else % if constraint == 0
        switch(state)
            case 1
                x = [2 3];
            case 2
                x = [1 2];
            case 3
                x = [1 2];
            case 4
                x = [1 2];
            case 5
                x = [1 2];
            case 6
                x = [1 3];
            case 7
                x = [3 4];
            case 12
                x = [3 4];
            case 13
                x = [3 4];
            case 15
                x = 3;
            case 18
                x = [3 4];
            case 19
                x = [3 4];
            case 21
                x = [3 4];
            case 22
                x = [1 2];
            case 23
                x = [1 2];
            case 24
                x = [1 3 4];
            case 25
                x = [2 3 4];
            case 26
                x = [1 2];
            case 27
                x = [1 4];
            case 30
                x = [3 4];
            case 31
                x = [3 4];
            case 36
                x = [3 4];
            case 37
                x = [3 4];
            case 42
                x = [3 4];
            case 43
                x = [3 4];
            case 48
                x = [3 4];
            case 49
                x = [2 4];
            case 50
                x = [1 2];
            case 51
                x = [1 2];
            case 52
                x = [1 2];
            case 53
                x = [1 2];
            case 54
                x = [1 4];
            otherwise
                x = [1 2 3 4];
        end
    end % end of if constraint == 1
end