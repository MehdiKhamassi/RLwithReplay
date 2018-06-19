function choice = drand01(p)
% This function consists in stochastically choosing among possible
% options depending on their respective probabilities
%
% INPUTS:
%     p is a vector of N elements representing probabilities of occurence of N concurrent options
%
% OUTPUTS:
%     choice is an integer containing the number of the chosen option
% 
%     created in 2005 at Okinawa Computational Neuroscience Course
%     by Mehdi Khamassi and Junichiro Yoshimoto
%     last modified 18 Jun 2018
%     by Mehdi Khamassi
%
%     correspondence: firstname (dot) lastname (at) upmc (dot) fr 

jjj = rand(1);
iii = 0;
cumul = 0;
letsContinue = true;
while letsContinue
    iii = iii + 1;
    cumul = cumul + p(iii);
    if ((jjj <= cumul)||(iii == length(p)))
        letsContinue = false;
    end
end
choice = iii;
