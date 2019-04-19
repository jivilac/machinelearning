#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:51:11 2019

@author: jivila
"""
'''
1. Read Data: For this assignment, assume input is CSV and write a function that
can read a csv into python. It’s ok to use an existing function that already exists
in python or pandas.

'''
import pandas as pd

def read_data_csv(csv):
    '''
    Description: Read a csv file
    '''
    data = pd.read_csv(csv)
    return data
    
