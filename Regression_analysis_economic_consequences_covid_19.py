##This is an example, to illustrate how to perform multiple regression analysis, using several packages.
##The data is based on IMF's estimates for GDP-growth correction in from 2019 to 2020, and indexes on government 
##stringency and economic support calculated by the project Our World in Data (URL: https://ourworldindata.org/)
##Note that the data was from 2020, and has not been updated. The results are therefore just an illustration and preliminary
##The framework for the analysis can be reproduced, but this should be done with better and updated data.
##IMF World Economic Outlook 2019: https://www.imf.org/en/Publications/WEO/Issues/2019/10/01/world-economic-outlook-october-2019
##IMF World Economic Outlook 2020: https://www.imf.org/en/Publications/WEO/Issues/2020/09/30/world-economic-outlook-october-2020
##Our World in Data - Stringency Index: https://ourworldindata.org/covid-stringency-index
##Our World in Data - Economic Support Index: https://ourworldindata.org/covid-income-support-debt-relief

import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
import scipy.stats as stats
from statsmodels.compat import lzip
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

## bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

## call in data set
World = pd.read_csv('https://raw.githubusercontent.com/albertovth/postulated-reality/master/data_economic_consequences_covid_EEA.csv')

World1=pd.DataFrame(World)


display(World1)
#%%
############################################################################################
#MULTIPLE REGRESSION
############################################################################################
# Regression with stringency index
print ("Association Between GDP estimate error and Stringency Index")
reg1 = smf.ols('GDP_estimate_error ~ Stringency_Index', data=World1).fit()
print (reg1.summary())
#%%
#Adding Economic support index
print ("Association Between GDP estimate error, Stringency Index and Economic support index")
reg2 = smf.ols('GDP_estimate_error ~ Stringency_Index + Economic_support_index', data=World1).fit()
print (reg2.summary())
#%%
#Adding ln(cases per capita)
print ("Association Between GDP estimate error, Stringency Index, Economic support index and cases per capita")
reg3 = smf.ols('GDP_estimate_error ~ Stringency_Index + Economic_support_index + ln_cases_per_capita_2020', data=World1).fit()
print (reg3.summary())

het_breuschpagan = sm.stats.diagnostic.het_breuschpagan(reg3.resid, reg3.model.exog, robust=True)

names = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']

het_b_results=lzip(names, het_breuschpagan)

print('Breuschpagan heteroscedasticity test results:')
print(het_b_results)

white_test = sm.stats.diagnostic.het_white(reg3.resid, reg3.model.exog)

white_results=lzip(names, white_test)

print('White heteroscedasticity test results:')
print(white_results)

y, X = dmatrices('GDP_estimate_error ~ Stringency_Index + Economic_support_index + ln_cases_per_capita_2020', data=World1, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)

with open('/Users/Alberto/Desktop/summary.txt', 'w') as fh:
    fh.write('Summary of OLS results for model with one variable (Stringency Index):'+ '\n')
    fh.write(reg1.summary().as_text() + '\n'+'\n')
    fh.write('Summary of OLS results for model with two variables (Stringency Index and Economic support index):'+ '\n')
    fh.write(reg2.summary().as_text()+'\n'+'\n')
    fh.write('Summary of OLS results for model with three variables (Stringency Index, Economic support index and covid-19-cases per capita):'+ '\n')
    fh.write(reg3.summary().as_text()+'\n'+'\n')
    fh.write('\n')
    fh.write('Breuschpagan heteroscedasticity test results for the model with trhee variables:'+'\n'+'\n')

    for t in het_b_results:
        fh.write(' '.join(str(s) for s in t) + '\n')
    fh.write('\n')
    fh.write('White heteroscedasticity test results for the model with trhee variables:'+'\n'+'\n')
    for p in white_results:
        fh.write(' '.join(str(q) for q in p) + '\n')
    fh.write('\n'+'\n')
    fh.write('Value Inflation Factor of variables in model'+'\n'+'\n')
    fh.write(vif.to_csv(sep=' ', index=False, header=False))

####################################################################################
# EVALUATING MODEL FIT
####################################################################################
#%%

#%%
# simple plot of residuals
stdres=pd.DataFrame(reg3.resid_pearson)
fig2 = plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print (fig2)
print(stdres)
#%%

# additional regression diagnostic plots
# For Stringency_Index 
fig3 = plt.figure(figsize=(12,8))
fig3 = sm.graphics.plot_regress_exog(reg3, 'Stringency_Index', fig=fig3)
print(fig3)

#%%
# For Economic_support_index
fig4 = plt.figure(figsize=(12,8))
fig4 = sm.graphics.plot_regress_exog(reg3, 'Economic_support_index', fig=fig4)
print(fig4)

#%%
# For ln_cases_per_capita_2020
fig5 = plt.figure(figsize=(12,8))
fig5 = sm.graphics.plot_regress_exog(reg3, 'ln_cases_per_capita_2020', fig=fig5)
print(fig5)


#%%
# leverage plot
fig6=sm.graphics.influence_plot(reg3, size=15)
print(fig6)

fig7=sm.qqplot(reg3.resid, stats.t,fit=True, line="45")
print(fig7)

plt.show()
