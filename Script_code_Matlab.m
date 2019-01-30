%% Stock Market Return Evaluation: An Approach Based on Vector Autoregressive  
%%                                                                            
%% ZAETTA Paul                                                                 
%%
clc;
clear all;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD THE DATASET AND TRANSFORMATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('Data_stock_market_return.mat');
[T,N] = size(index_prices_all);

%-------------------------------------------------------------------------%
% We transform prices into log-returns                                    %
%-------------------------------------------------------------------------%

index_log_prices_all = zeros(length(index_prices_all),N-1);
index_log_prices_all(:,1) = log(index_prices_all(:,2));
index_log_prices_all(:,2) = log(index_prices_all(:,3));
index_log_prices_all(:,3) = log(index_prices_all(:,4));

index_returns_all = zeros(T-1, N-1);
for j = 2:N
    for i = 2:T
        index_returns_all(i-1,j-1) = log(index_prices_all(i,j)/index_prices_all(i-1,j))*100;
    end
end
clear i j; 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DESCRIPTIVE SATATISTICS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure(1)
standardized_prices = bsxfun(@rdivide,bsxfun(@minus,index_prices_all,mean(index_prices_all)),std(index_prices_all));
plot(standardized_prices(:,2), 'b-'); hold on
plot(standardized_prices(:,3), 'r-'); hold on
plot(standardized_prices(:,4), 'y-');
xlabel('Obeservation');
ylabel('Standardized prices');
legend('S&P 500 evolution', 'CAC 40 evolution', 'Nikkei 225 evolution');
title('Stock Market Indices performance');
axis tight


figure(2)
subplot(3, 1, 1)
plot(index_returns_all(:,1));
ylabel('Returns(%)');
title('S&P 500 returns evolution');
axis tight

subplot(3, 1, 2)
plot(index_returns_all(:,2));
ylabel('Returns(%)');
title('CAC 40 returns evolution');
axis tight

subplot(3, 1, 3)
plot(index_returns_all(:,3));
xlabel('Obeservation');
ylabel('Returns(%)');
title('Nikkei 225 returns evolution');
axis tight


%%
%-------------------------------------------------------------------------%
% Unit root test on returns (Augmented Dickey-Fuller)                     %
%                                                                         %
% 2 steps:                                                                %
% - study if the t-ratios are statistically significantly                 %                                     
% - check if the residuals are not correlated                             %                
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% We check if the (t-1) ratios is statistically significantly             %
%-------------------------------------------------------------------------%

adf_GSPC = adfreg(index_returns_all(:,1), 0);
t_sig1_GSPC = adf_GSPC.tsig(2,1);

adf_C40 = adfreg(index_returns_all(:,2), 0);
t_sig1_C40 = adf_C40.tsig(2,1);

adf_N225 = adfreg(index_returns_all(:,3), 0);
t_sig1_N225 = adf_N225.tsig(2,1);

%-----------------------------------------------------------------------------------------%
% The function already implemented are an alternative to determine the pValue (ADF test)  %
%-----------------------------------------------------------------------------------------%

%[h,pValue] = adftest(index_returns_all(:,1), 'lags', 1);
%[h,pValue] = adftest(index_returns_all(:,2), 'lags', 1);
%[h,pValue] = adftest(index_returns_all(:,3), 'lags', 1);


%%
%--------------------------------------------------------------------------------%
% Ljung Box Test (autocorrelation test on the estimated residuals (from returns) %
%--------------------------------------------------------------------------------%

LB_GSPC = ljung_box(adf_GSPC.resid,1);
LB_Q_GSPC = LB_GSPC.LB_stat;
LB_pValue_GSPC = LB_GSPC.pValue;

LB_C40 = ljung_box(adf_C40.resid,1);
LB_Q_C40 = LB_C40.LB_stat;
LB_pValue_C40 = LB_C40.pValue;

LB_N225 = ljung_box(adf_N225.resid,1);
LB_Q_N225 = LB_N225.LB_stat;
LB_pValue_N225 = LB_N225.pValue;

%----------------------------------------------------------------------------------------------%
% The function already implemented are an alternative to determine the pValue (Ljung-Box test) %
%----------------------------------------------------------------------------------------------%

%[h, pValue] = lbqtest(adf_GSPC.resid)
%[h, pValue] = lbqtest(adf_C40.resid)
%[h, pValue] = lbqtest(adf_N225.resid(1:300))
%[h2] = lbqtest(adf_N225.resid.^2)
%[h3] = archtest(adf_N225.resid)


%%
%-------------------------------------------------------------------------%
% We check the autocorrelation functions for the residuals of each series %
%-------------------------------------------------------------------------%

figure(3)
subplot(3,1,1)
epsi1 = adf_GSPC.resid;
H = 10;
gama_emp = autocov_emp_vec(epsi1, H);
rho_emp = gama_emp / var(epsi1);
bar([0:H], [1 rho_emp], 'r');
title('Autocorrelation function for S&P 500 residuals');
axis tight

subplot(3,1,2)
epsi2 = adf_C40.resid;
H = 10;
gama_emp = autocov_emp_vec(epsi1, H);
rho_emp = gama_emp / var(epsi2);
bar([0:H], [1 rho_emp], 'r');
title('Autocorrelation function for CAC 40 residuals');
axis tight

subplot(3,1,3)
epsi3= adf_N225.resid;
H = 10;
gama_emp = autocov_emp_vec(epsi3, H);
rho_emp = gama_emp / var(epsi3);
bar([0:H], [1 rho_emp], 'r');
title('Autocorrelation function for Nikkei 225 residuals');
axis tight


%%
%-------------------------------------------------------------------------%
% Clear previous useless variables                                        %
%-------------------------------------------------------------------------%
 
clear T N H;
clear standardized_prices;
clear adf_GSPC adf_C40 adf_N225 
clear t_sig1_GSPC t_sig1_C40  t_sig1_N225; 
clear epsi1 epsi2 epsi3 gama_emp rho_emp; 
clear LB_Q_GSPC LB_Q_C40 LB_Q_N225 ;
clear LB_C40 LB_GSPC LB_N225;
clear LB_pValue_GSPC LB_pValue_C40 LB_pValue_N225;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VECTOR AUTOREGRESSIVE MOVING AVERAGE VARMA(1,1) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T, N] = size(index_returns_all);

Y = index_returns_all(2:end,:);
Z = [ones(T-1, 1) index_returns_all(1:T-1, :)];
coeff_OLS = (Z'*Z)\(Z'*Y);
res_OLS = Y - Z*coeff_OLS;

Y2 = Y(2:end,:);
Z2 = [Z(2:end,:) res_OLS(1:end-1, :)];

coeff_VARMA_OLS = (Z2'*Z2)\(Z2'*Y2);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FORECATING RETURNS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

coeff_VARMA_OLS = coeff_VARMA_OLS';
c = coeff_VARMA_OLS(:,1);
coeff_AR = coeff_VARMA_OLS(:, 2:N+1);

horizon = 7;

returns_forecasted = [index_returns_all(end,:)' zeros(N,horizon)];
for h = 2:horizon+1
    returns_forecasted(:,h) = c + coeff_AR*returns_forecasted(:,h-1);
end


%%
%-------------------------------------------------------------------------%
% Plot forecasted returns on the same scale                               %
%-------------------------------------------------------------------------%

figure(4)
plot(returns_forecasted(1,1:4),'r-*'); hold on
plot(returns_forecasted(2,1:4),'b-*'); hold on 
plot(returns_forecasted(3,1:4),'m-*');
legend('forecasted S&P 500 returns', 'forecasted CAC 40 returns', 'forecasted Nikkei 225 returns');
xlabel('Time');
ylabel('Returns (%)');
title('Forcasted Returns');
axis ([1 4 -1 0.5 ])
 

%-------------------------------------------------------------------------%
% FORECATING RETURNS with CONFIDENCE INTERVAL at 90%                      %
%-------------------------------------------------------------------------%

CI_1 = [returns_forecasted(1,1:2);returns_forecasted(1,1:2);returns_forecasted(1,1:2)];
CI_2 = [returns_forecasted(2,1:2);returns_forecasted(2,1:2);returns_forecasted(2,1:2)];
CI_3 = [returns_forecasted(3,1:2);returns_forecasted(3,1:2);returns_forecasted(3,1:2)];

SP500_exp = c(1,1) + index_returns_all * coeff_AR(1,:)';
CAC40_exp = c(2,1) + index_returns_all * coeff_AR(2,:)';
NIKKEI225_exp = c(3,1) + index_returns_all * coeff_AR(3,:)';

STD_1 = std(index_returns_all(:,1)-SP500_exp);
STD_2 = std(index_returns_all(:,2)-CAC40_exp);
STD_3 = std(index_returns_all(:,3)-NIKKEI225_exp);

CI_1(1,2) = CI_1(3,2) + STD_1 * 1.645;
CI_1(2,2) = CI_1(3,2) - STD_1 * 1.645;
CI_2(1,2) = CI_2(3,2) + STD_2 * 1.645;
CI_2(2,2) = CI_2(3,2) - STD_2 * 1.645;
CI_3(1,2) = CI_3(3,2) + STD_3 * 1.645;
CI_3(2,2) = CI_3(3,2) - STD_3 * 1.645;

figure(7)
subplot(3,1,1)
plot(returns_forecasted(1,1:2),'r-*'); hold on
plot(2,CI_1(1,2),'rx'); hold on
plot(2,CI_1(2,2),'rx'); hold on
ylabel('Returns (%)');
title('S&P 500');
axis ([1 2.2 -4 4]);

subplot(3,1,2)
plot(returns_forecasted(2,1:2),'b-*'); hold on
plot(2,CI_2(1,2),'bx'); hold on
plot(2,CI_2(2,2),'bx'); hold on
ylabel('Returns (%)');
title('CAC 40');
axis ([1 2.2 -4 4]);

subplot(3,1,3)
plot(returns_forecasted(3,1:2),'m-*'); hold on
plot(2,CI_3(1,2),'mx'); hold on
plot(2,CI_3(2,2),'mx'); hold on
ylabel('Returns (%)');
title('Nikkei 225');
axis ([1 2.2 -4 4]);


%%
%-------------------------------------------------------------------------%
% Clear previous useless variables                                        %
%-------------------------------------------------------------------------%

clear c h T N horizon Y Y2 Z Z2;
clear coeff_AR coeff_OLS coeff_VARMA_OLS;
clear res_OLS returns_forecasted;
clear CI_1 CI_2 CI_3 STD_1 STD_2 STD_3;
clear CAC40_exp SP500_exp NIKKEI225_exp;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FORECATING RETURNS: learning sample/testing sample (80%/20%) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-------------------------------------------------------------------------%
% VECTOR AUTOREGRESSIVE MOVING AVERAGE VARMA(1,1) on 80% of the sample    %
%-------------------------------------------------------------------------%

learning_sample_length = round(0.8*length(index_returns_all));
testing_sample_length = length(index_returns_all) - learning_sample_length;


[T, N] = size(index_returns_all(1:learning_sample_length,:));

Y = index_returns_all(2:T,:);
Z = [ones(T-1, 1) index_returns_all(1:T-1, :)];
coeff_OLS = (Z'*Z)\(Z'*Y);
res_OLS = Y - Z*coeff_OLS;

Y2 = Y(2:end,:);
Z2 = [Z(2:end,:) res_OLS(1:end-1, :)];


%%
%-------------------------------------------------------------------------%
% FORECATING RETURNS                                                      %
%-------------------------------------------------------------------------%

coeff_VARMA_OLS = (Z2'*Z2)\(Z2'*Y2);

coeff_VARMA_OLS = coeff_VARMA_OLS';
c = coeff_VARMA_OLS(:,1);
coeff_AR = coeff_VARMA_OLS(:, 2:N+1);

horizon = testing_sample_length;

returns_forecasted = [index_returns_all(end,:)' zeros(N,horizon)];
for h = 2:horizon+1
    returns_forecasted(:,h) = c + coeff_AR*returns_forecasted(:,h-1);
end


%%
%-------------------------------------------------------------------------%
% Plot forcasted returns on a subsample                                   %
%-------------------------------------------------------------------------%

index_returns_all = index_returns_all';
returns_forecasted_end = [index_returns_all(:,1:learning_sample_length-1) returns_forecasted];


figure(8)
subplot(3,1,1)
plot(returns_forecasted_end(1,end-testing_sample_length-5:end-testing_sample_length+3),'r-'); hold on
plot(index_returns_all(1,end-testing_sample_length-5:end-testing_sample_length+3), 'b-');
legend('forecasted S&P 500 returns', 'true S&P 500 returns');
ylabel('Returns (%)');
axis ([1 8 -3 3]);

subplot(3,1,2)
plot(returns_forecasted_end(2,end-testing_sample_length-5:end-testing_sample_length+3),'r-'); hold on
plot(index_returns_all(2,end-testing_sample_length-5:end-testing_sample_length+3), 'b-');
legend('forecasted CAC 40 returns', 'true CAC 40 returns');
ylabel('Returns (%)');
axis ([1 8 -3 3]);

subplot(3,1,3)
plot(returns_forecasted_end(3,end-testing_sample_length-5:end-testing_sample_length+3),'r-'); hold on
plot(index_returns_all(3,end-testing_sample_length-5:end-testing_sample_length+3), 'b-');
legend('forecasted Nikkei 225 returns', 'true Nikkei 225 returns');
ylabel('Returns (%)');
axis ([1 8 -3 3]);


%%
%-------------------------------------------------------------------------%
% Clear previous useless variables                                        %
%-------------------------------------------------------------------------%

clear c h T N horizon Y Y2 Z Z2;
clear learning_sample_length testing_sample_length;
clear coeff_AR coeff_OLS coeff_VARMA_OLS;
clear res_OLS returns_forecasted returns_forecasted_end;
