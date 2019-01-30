# Folder structure

## Data_stock_market_return.mat

It contains the final data set. The observations of the stock market indices come from `https://fr.finance.yahoo.com/`. The data set for this study consists of 3514 valid data points for each series (from 16th December 2002 to 14th December 2017).

## Functions_code_Matlab

It contains the other functions that were used during this study. The adfreg function is used for the Augmented Dickey-Fuller test (it checks whether a unit root is present in time series sample). The autocov_emp_vec function allows us to compute the empirical autocovariance (with vectors). The dfcrit function corresponds to the critical Dickey-Fuller values. Finally, the ljung_box is used for the Ljung-Box test (this function is a type of statistical test of wether any of a group of autocorrelations of a time series are different from zero).

## Script_code_Matlab

It contains the main script, which allow to run all the results obtained and the graphic illustrations, which are present in the project. 

## Stock_Market_Return_Evaluation_Project.pdf

This file contains the final project (in PDF format), which is "Stock Market Return Evaluation: An Approach Based on Vector Autoregressive".
