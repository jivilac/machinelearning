#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:50:35 2019

@author: jivila
"""

import read_data as rd
import explore_data as ed
import clean_data as cd
import create_features as cf
import build_models as bm
import eval_class as ec
import numpy as np

#Importing data and l
data = rd.read_data_csv("credit-data.csv")
data_id = 'PersonID'
y='SeriousDlqin2yrs'
float_columns = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome', 'DebtRatio' ]
cat_columns =['SeriousDlqin2yrs', 'zipcode', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse' , 'NumberOfDependents']
cols_to_transform = ['NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTime30-59DaysPastDueNotWorse']
X_col = float_columns.copy()
#creating variables acording type of data

#Describing data
data_describe = ed.describe_data(data,[.01, .20, .40, .60, .80, .99])
missing = ed.missing_values_table(data)
correlation = ed.correlation(data)
ed.correlation_graph(data)
fre_tables = ed.freq_tables(data, cat_columns, data_id)

#Y values
data[y].value_counts()
ed.graph_dummy(y,data)


#Replacing missing data by the mean
data['missing_dependants'] = data['NumberOfDependents'].notna()
data['missing_income'] = data['MonthlyIncome'].notna()
data_clean = cd.replace_missing(data)

#comment: there is a issue with debt ratio data when the income is 0 or null this provokes that the debt ratio are to high
data_clean = data_clean[data_clean['MonthlyIncome'] != 0]
data_Ratio = data_clean.groupby(data['missing_income'])['DebtRatio'].describe()
data_clean = data_clean[data_clean['missing_income'] != False]
data_clean = data_clean[data_clean['NumberOfTime60-89DaysPastDueNotWorse'] != 98]
data_clean = data_clean[data_clean['NumberOfTime60-89DaysPastDueNotWorse'] != 96]
#creating features

ed.histogram(data_clean.NumberOfOpenCreditLinesAndLoans)
data_clean['NumberOfOpenCreditLinesAndLoans'].value_counts()
X_col.append('NumberOfOpenCreditLinesAndLoans')
# NumberOfTimes90DaysLate
ed.histogram(data_clean.NumberOfTimes90DaysLate)
data_clean.NumberOfTimes90DaysLate.value_counts()
data_clean.NumberOfTimes90DaysLate.describe()
data_clean['d_90DaysLate'] = np.where(data_clean['NumberOfTimes90DaysLate']>1, 1, 0)
X_col.append('d_90DaysLate')


#NumberRealEstateLoansOrLines
ed.histogram(data_clean.NumberRealEstateLoansOrLines)
data_clean['NumberRealEstateLoansOrLines'].value_counts()
data_clean['d_nrealstates1'] = np.where(data_clean['NumberRealEstateLoansOrLines'] == 1, 1, 0)
data_clean['d_nrealstates2'] = np.where(data_clean['NumberRealEstateLoansOrLines'] == 2, 1, 0)
data_clean['d_nrealstates3'] = np.where(data_clean['NumberRealEstateLoansOrLines'] > 2, 1, 0)


X_col.append('d_nrealstates1')
X_col.append('d_nrealstates2')
X_col.append('d_nrealstates3')

#NumberOfTime60-89DaysPastDueNotWorse
ed.histogram(data_clean['NumberOfTime60-89DaysPastDueNotWorse'])
data_clean['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()
data_clean['NumberOfTime60-89DaysPastDueNotWorse'].describe()
data_clean['d_60-89DaysPastDueNotWorse'] = np.where(data_clean['NumberOfTime60-89DaysPastDueNotWorse']>1, 1, 0)

X_col.append('d_60-89DaysPastDueNotWorse')
#NumberOfTime30-59DaysPastDueNotWorse
ed.histogram(data_clean['NumberOfTime30-59DaysPastDueNotWorse'])
data_clean['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()
data_clean['NumberOfTime30-59DaysPastDueNotWorse'].describe()
data_clean['d_Time30-59_1'] = np.where(data_clean['NumberRealEstateLoansOrLines'] == 1, 1, 0)
data_clean['d_Time30-59_2'] = np.where(data_clean['NumberRealEstateLoansOrLines'] == 2, 1, 0)
data_clean['d_Time30-59_3'] = np.where(data_clean['NumberRealEstateLoansOrLines'] > 2, 1, 0)
X_col.append('d_Time30-59_1')
X_col.append('d_Time30-59_2')
X_col.append('d_Time30-59_3')

#zipcode
df_zipcode= pd.get_dummies(data_clean['zipcode'])

#creating data sets

X = data_clean[X_col]
X = X.join(df_zipcode)
Y=data_clean['SeriousDlqin2yrs']
