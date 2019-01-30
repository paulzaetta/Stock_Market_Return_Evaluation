function gama = autocov_emp_vec(X,H)
%--------------------------------------------------------------------------
% This function computes the empirical autocovariance of order H
%--------------------------------------------------------------------------
% INPUTS: 
% X : the serie
% H : the number of lags
%--------------------------------------------------------------------------
% Copyright P.ZAETTA 2017
%--------------------------------------------------------------------------

T = length(X);

gama = zeros(1, H);

for k = 1:H
gama(k) = mean(X(k+1:T).*X(1:T-k)) - mean(X)*mean(X(k+1:T));
end
end
