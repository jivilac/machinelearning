#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:59:14 2019

@author: jivila
"""
'''
2. Explore Data: You can use the code you wrote for assignment 1 here to generate
distributions of variables, correlations between them, find outliers, and data
summaries.
'''
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as pl


def missing_values_table(df):
        missing_v_percent = 100 * df.isnull().sum() / len(df)
        missing_v = df.isnull().sum()
        missing_v_table = pd.concat([missing_v, missing_v_percent], axis=1)
        missing_v_table_ren_columns = missing_v_table.rename(
        columns = {0 : 'Nan Val', 1 : '% of Total Values'})
        missing_v_table_ren_columns = missing_v_table_ren_columns[
            missing_v_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(missing_v_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return missing_v_table_ren_columns

def describe_data(df, distribution = [.20, .40, .60, .80] ): 
    perc = distribution
    include =['object', 'float', 'int'] 
    desc = df.describe(percentiles = perc, include = include) 
    return desc 

def correlation(df):
    corr =  df.corr()
    return corr    
    
def correlation_graph(df):
    corr =  df.corr()
    sns.heatmap(corr)
    return  pl.show()

def histogram(columns):
    '''
    Note it requiere not missing column
    '''
    sns.set_style('darkgrid')
    return sns.distplot(columns)

def freq_tables(df, columns, ids):
    '''
    Create freq tables for list of columns
    '''
    dic= {}
    for i in columns:
        temp = df.groupby(i)[ids].nunique()
        dic[i] =temp
    return dic

def describe_y(df, col_y):
    '''
    NOT WORKING
    '''
    d_y = df[col_y].value_count()
    return d_y


def graph_dummy(col, df):
    sns.countplot(x=col, data = df,palette='hls')
    return pl.show()




