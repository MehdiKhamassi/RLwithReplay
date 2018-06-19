function x = isForward(state1, state2)
% This function returns 1 is the sequence state1->state2 corresponds to a
% forward move in the multiple-T-maze task, -1 if it is backward, and 0
% otherwise
%
% INPUTS:
%     state1 is an identification number (between 1 and 54)
%     state2 is an identification number (between 1 and 54)
%
% OUTPUTS:
%     x is 1 if forward sequence, -1 if backward, 0 otherwise
% 
%     created 21 Sept 2017
%     by Mehdi Khamassi
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

    %% x = 1 if forward -1 if backward 0 otherwise
    switch(state1)
        case 1
            if (state2 == 2)
                x = 1;
            else
                if (state2 == 7)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 2
            if (state2 == 3)
                x = 1;
            else
                if (state2 == 1)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 3
            if (state2 == 4)
                x = 1;
            else
                if (state2 == 2)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 4
            if (state2 == 5)
                x = 1;
            else
                if (state2 == 3)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 5
            if (state2 == 6)
                x = 1;
            else
                if (state2 == 4)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 6
            if (state2 == 12)
                x = 1;
            else
                if (state2 == 5)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 7
            if (state2 == 1)
                x = 1;
            else
                if (state2 == 13)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 12
            if (state2 == 18)
                x = 1;
            else
                if (state2 == 6)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 13
            if (state2 == 7)
                x = 1;
            else
                if (state2 == 19)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 15
            if (state2 == 21)
                x = 1;
            else
                if (state2 == 27)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 18
            if (state2 == 24)
                x = 1;
            else
                if (state2 == 12)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 19
            if (state2 == 13)
                x = 1;
            else
                if (state2 == 25)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 21
            if ((state2 == 15)||(state2 == 27))
                x = 1;
            else
                if (state2 == 22)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 22
            if (state2 == 21)
                x = 1;
            else
                if (state2 == 23)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 23
            if (state2 == 22)
                x = 1;
            else
                if (state2 == 24)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 24
            if (state2 == 23)
                x = 1;
            else
                if ((state2 == 18)||(state2 == 30))
                    x = -1;
                else
                    x = 0;
                end
            end
        case 25
            if ((state2 == 19)||(state2 == 31))
                x = 1;
            else
                if (state2 == 26)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 26
            if (state2 == 25)
                x = 1;
            else
                if (state2 == 27)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 27
            if (state2 == 26)
                x = 1;
            else
                if (state2 == 21)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 30
            if (state2 == 24)
                x = 1;
            else
                if (state2 == 36)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 31
            if (state2 == 37)
                x = 1;
            else
                if (state2 == 25)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 36
            if (state2 == 30)
                x = 1;
            else
                if (state2 == 42)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 37
            if (state2 == 43)
                x = 1;
            else
                if (state2 == 31)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 42
            if (state2 == 36)
                x = 1;
            else
                if (state2 == 48)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 43
            if (state2 == 49)
                x = 1;
            else
                if (state2 == 37)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 48
            if (state2 == 42)
                x = 1;
            else
                if (state2 == 54)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 49
            if (state2 == 50)
                x = 1;
            else
                if (state2 == 43)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 50
            if (state2 == 51)
                x = 1;
            else
                if (state2 == 49)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 51
            if (state2 == 52)
                x = 1;
            else
                if (state2 == 50)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 52
            if (state2 == 53)
                x = 1;
            else
                if (state2 == 51)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 53
            if (state2 == 54)
                x = 1;
            else
                if (state2 == 52)
                    x = -1;
                else
                    x = 0;
                end
            end
        case 54
            if (state2 == 48)
                x = 1;
            else
                if (state2 == 53)
                    x = -1;
                else
                    x = 0;
                end
            end
        otherwise
            x = 0;
    end
end