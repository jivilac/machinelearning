#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:45:01 2019

@author: jivila
"""
'''
3. Pre-Process Data: For this assignment, you can limit this to filling
in missing values for the variables that have missing values. You can use any
simple method to do it (use mean or median to fill in missing values).


'''

import pandas as pd
import numpy as np

def replace_missing(df, mean=True):
    if mean:
        rv = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return rv

def remove_outlier(df,col):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    dout = df.copy()
    for i in col:
        dout[col]= df[((df < (Q1 - 1.5 * IQR[col])) |(df > (Q3 + 1.5 * IQR[col]))).any(axis=1)]
    return dout

def drop_na(df):
    rv = df.dropna(inplace = True)
    return rv
#
