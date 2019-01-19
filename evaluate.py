""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_recall_curve,recall_score,precision_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', '-T', default=0.1, type=float, help="threshold")
args = parser.parse_args()
# from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

def evaluate(labels, scores, directory,metric='roc'):
    if metric == 'roc':
        # print(directory)
        return roc(labels, scores, directory, plot=True )
    elif metric == 'auprc':
        return auprc(labels, scores, directory,plot=True)
    elif metric == 'f1_score':
        return f1(labels, scores, directory, plot=True)
    elif metric == 'recall':
        return recall(labels, scores, directory,plot=True)
    elif metric == 'precision':
        return precision(labels, scores, directory, plot=True)
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, directory, plot=True):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()



    return roc_auc_score(labels, scores)

def auprc(labels, scores, directory, plot=True):

    ap = average_precision_score(labels, scores)
    return ap
def f1(labels, scores, directory, plot=False):
    threshold = args.threshold
    y_pre = []
    for i in range(0,len(scores)):
        if scores[i][0]> scores[i][1]:
            y_pre.append(1)
        else:
            y_pre.append(0)
    f1 = f1_score(labels, y_pre)
    # print(f1)
    return f1

def recall(labels, scores, directory,plot=False):

    # ap = average_precision_score(labels, scores)
    # return ap
    scores_sorted = sorted(scores)
    lenth = int(len(scores_sorted) * 0.8)
    threshold = scores_sorted[lenth]
    scores = np.array(scores)
    for i in range(0,len(scores)):
        if scores[i] >= threshold:
            scores[i] = 1
        else:
            scores[i] = 0
    recall = recall_score(labels, scores)
    # print(recall)
    return recall

def precision(labels, scores, directory,plot=False):

    # ap = average_precision_score(labels, scores)
    # return ap
    scores_sorted = sorted(scores)
    lenth = int(len(scores_sorted) * 0.8)
    threshold = scores_sorted[lenth]
    scores = np.array(scores)
    for i in range(0,len(scores)):
        if scores[i] >= threshold:
            scores[i] = 1
        else:
            scores[i] = 0
    precision = precision_score(labels, scores)
    # print(precision)
    return precision