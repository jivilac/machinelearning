#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:14:44 2019

@author: jivila
"""

'''
5. Build Classifier: For this assignment, select any classifier you feel comfortable
with (Logistic Regression or Decision Trees)

'''
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns



def hyper_param_select(models, param_grid, clfs, x_train,y_train, sc_metric='precision' ):
    rv=dict()
    for model in models:
        clf = GridSearchCV(clfs[model],param_grid[model], cv=5,scoring = sc_metric)
        clf.fit(x_train,y_train)
        rv[model]=clf.best_params_
    return rv

    
def gen_train_test_data(X,y,size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0)
    return X_train, X_test, y_train, y_test


def predict_result(X_test, y_test, model):
    y_pred = model.predict(X_test)
    print('Accuracy of model regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    rv = confusion_matrix(y_test, y_pred)
    return rv

def auc_roc_graph(X_test, y_test, model):
    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Model (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(str(model)+'_ROC')
    return plt.show()

def rolling_op(df,t ):
    df['month'] = df[t].dt.month
    df['year'] = df[t].dt.year
    df['semester'] = np.where(df[t].dt.month <=6, 1, 2)
    df['id'] = df.groupby(['year', 'semester']).grouper.group_info[0]
    rv=[]
    for i in range(df.id.unique().max()):
        train = df.loc[df['id'] <= i]
        train = train.drop('month', axis=1)
        train = train.drop('year', axis=1)
        train = train.drop('semester', axis=1)
        train = train.drop('id', axis=1)
        train = train.drop(t, axis=1)
        test = df.loc[df['id'] == i + 1]
        test = test.drop('month', axis=1)
        test = test.drop('year', axis=1)
        test = test.drop('semester', axis=1)
        test = test.drop('id', axis=1)
        test = test.drop(t, axis=1)
        tup = (train , test)
        rv.append(tup)
    return rv

def rolling_window(y,x,t):
    y_cutted = rolling_op(y, t )
    x_cutted = rolling_op(x, t )
    rv =dict()
    for i in range(len(y_cutted)):
        y_train, y_test = y_cutted[i]
        x_train, x_test = x_cutted[i]
        rv[i] = (y_train, y_test ,x_train, x_test)
    return rv

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true, preds_at_k)
    return recall


def f1_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    f1 = f1_score(y_true, preds_at_k)
    return f1

def accuracy_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    rv = accuracy_score(y_true, preds_at_k)
    return rv



def clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50','r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50','f1_at_1', 'f1_at_2', 'f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30', 'f1_at_50','acc1_at_1', 'acc_at_2', 'acc_at_5', 'acc_at_10', 'acc_at_20', 'acc_at_30', 'acc_at_50'))
    for n in range(1, 2):
        # create training and valdation sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(x_train.values, y_train.values).predict_proba(x_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs,y_test.values), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                      ]
                    
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df




def model_selector(X,y,param_grid,clfs,t,models_to_run,metric='precision'):
    tyt_data = rolling_window(y,X,t)
    y_train, y_test ,x_train, x_test = tyt_data[0]
    #param = hyper_param_select(models_to_run, param_grid, clfs, x_train,y_train, sc_metric=metric )
    results_df =  pd.DataFrame(columns=('test_data','model_type','clf', 'parameters', 'auc-roc','p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50','r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20', 'r_at_30', 'r_at_50','f1_at_1', 'f1_at_2', 'f1_at_5', 'f1_at_10', 'f1_at_20', 'f1_at_30', 'f1_at_50','acc1_at_1', 'acc_at_2', 'acc_at_5', 'acc_at_10', 'acc_at_20', 'acc_at_30', 'acc_at_50'))
    for i,data in enumerate(tyt_data):
        y_train, y_test ,x_train, x_test = tyt_data[i]
        result = clf_loop(models_to_run, clfs, param_grid, x_train, x_test, y_train, y_test)
        results_df = results_df.append(result)
        results_df['test_data'] = results_df['test_data'].fillna('test'+str(i+1)) 
    return results_df
        