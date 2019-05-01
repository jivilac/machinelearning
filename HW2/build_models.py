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

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


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
    plt.plot(fpr, tpr, label='Model (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(str(model)+'_ROC')
    return plt.show()