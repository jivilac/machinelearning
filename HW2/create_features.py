#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:14:43 2019

@author: jivila
"""

'''
4. Generate Features/Predictors: For this assignment, you should write one function
that can discretize a continuous variable and one function that can take a
categorical variable and create binary/dummy variables from it. Apply them to at
least one variable each in this data.

'''
import pandas as pd

def get_dummy(cat_vars, df):
    data1=df
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        data1=data1.join(cat_list)
    return data1


def x_creator(x):
    for i in x:
        x.append(i)
    return x
