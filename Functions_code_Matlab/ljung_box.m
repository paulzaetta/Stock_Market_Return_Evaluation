function result = ljung_box(Y,h)
%--------------------------------------------------------------------------
% ljung_box_test: this function is a type of statistical test of wether any
% of a group of autocorrelations of a time series are different from zero.
%--------------------------------------------------------------------------
% INPUTS: 
% Y : the serie (the vector of residuals)
% h : the number of autocorrelation lags
%--------------------------------------------------------------------------
% OUTPUTS: 
% result.rho_hat         : autocorrelation coefficients (estimated by OLS)
% result.LB_stat         : Ljung-Box statistic
% result.pValue          : Ljung-Box pValue
%--------------------------------------------------------------------------
% Copyright P.ZAETTA 2017
%--------------------------------------------------------------------------

T = length(Y);

rho_k = autocov_emp_vec(Y,h)./var(Y);
a_k = zeros(h,1);
for i = 1:h
    a_k(i) = (rho_k(i)^2)/(T-i);
end

LB_stat = T*(T+2)*sum(a_k);
pValue = 1 - chi2cdf(LB_stat, h);

result.rho = rho_k;
result.LB_stat = LB_stat;
result.pValue = pValue;
end

