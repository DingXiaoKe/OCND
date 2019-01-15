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

def evaluate(labels, scores, directory, epoch, inlier_class,metric='roc'):
    if metric == 'roc':
        # print(directory)
        return roc(labels, scores, directory, epoch, inlier_class, plot=True )
    elif metric == 'auprc':
        return auprc(labels, scores, directory, epoch, inlier_class, plot=True)
    elif metric == 'f1_score':
        return f1(labels, scores, directory, epoch, inlier_class, plot=True)
    elif metric == 'recall':
        return recall(labels, scores, directory, epoch, inlier_class, plot=True)
    elif metric == 'precision':
        return precision(labels, scores, directory, epoch, inlier_class, plot=True)
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, directory, epoch, inlier_class, plot=True):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    #fpr, tpr, _ = roc_curve(labels, scores)
    #roc_auc = auc(fpr, tpr)

    ## Equal Error Rate
    #eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    #if plot:
    #    plt.figure()
    #    lw = 2
    #    plt.step(fpr, tpr, color='darkorange',lw=lw, alpha=0.2, where='post')
    #    plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
    #    plt.xlabel('fpr')
    #    plt.ylabel('tpr')
    #    plt.ylim([0.0, 1.05])
    #    plt.xlim([0.0, 1.0])
    #    plt.title('ROC curve : AUC=%0.4f' 
    #                        %(roc_auc))
    #    if not os.path.exists(directory):
    #        os.makedirs(directory)
    #    plt.savefig(directory +str(inlier_class)+ '_'+ str(epoch)+ '_'+ 'auroc.png')
    #    plt.close()
    return roc_auc_score(labels, scores)

def auprc(labels, scores, directory,  epoch, inlier_class, plot=True):

    ap = average_precision_score(labels, scores)
    return ap
    #precision = dict()
    #recall = dict()
    #prc_auc = dict()

    #precision, recall, thresholds = precision_recall_curve(labels, scores)
    #prc_auc = auc(recall, precision)

    #if plot:
    #    plt.figure()
    #    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    #    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #    plt.xlabel('Recall')
    #    plt.ylabel('Precision')
    #    plt.ylim([0.0, 1.05])
    #    plt.xlim([0.0, 1.0])
    #    plt.title('Precision-Recall curve: AUC=%0.4f' 
    #                        %(prc_auc))
    #    if not os.path.exists(directory):
    #        os.makedirs(directory)
    #    plt.savefig(directory +str(inlier_class)+ '_'+ str(epoch)+ '_'+ 'auprc.png')
    #    plt.close()

    #return prc_auc

def f1(labels, scores, directory,  epoch, inlier_class, plot=False):

    # ap = average_precision_score(labels, scores)
    # return ap
    #scores_sorted = sorted(scores)
    #lenth = int(len(scores_sorted) * 0.8)
    #threshold = scores_sorted[lenth]
    #scores = np.array(scores)
    #print(min(scores))
    #print(max(scores))
    threshold = args.threshold
    # print(len(scores))
    y_pre = []
    for i in range(0,len(scores)):
        #if scores[i] >= threshold:
         #   scores[i] = 1
        #else:
         #   scores[i] = 0
        if scores[i][0]> scores[i][1]:
            y_pre.append(1)
        else:
            y_pre.append(0)
    f1 = f1_score(labels, y_pre)
    # print(f1)
    return f1

def recall(labels, scores, directory,  epoch, inlier_class, plot=False):

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

def precision(labels, scores, directory,  epoch, inlier_class, plot=False):

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

def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)
