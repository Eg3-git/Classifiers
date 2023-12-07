import csv
import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from scipy.spatial.distance import euclidean
import math
from joblib import dump
from feature_extraction import extract
from fastdtw import fastdtw

dir1a = "3USER_10TASKS/index/ABC1/U1_abc.csv"
dir1b = "3USER_10TASKS/robot-endeffector/ABC1/U1_ABC-pos3.mat"


dir2a = "3USER_10TASKS/index/ABC1/U2_abc.csv"
dir2b = "3USER_10TASKS/robot-endeffector/ABC1/U2_ABC-pos3.mat"


dir3a = "3USER_10TASKS/index/ABC1/U3_abc.csv"
dir3b = "3USER_10TASKS/robot-endeffector/ABC1/U3_ABC-pos3.mat"


def dtw(train, test, weights=[1 for _ in range(17)]):
    sum_n = []
    for n in train:
        sum_k = 0
        for k in range(17):
            feature_weight = weights[k]
            train_seq = [o[k] for o in n]
            test_seq = [o[k] for o in test]


            dtw_distance, warp_path = fastdtw(test_seq, train_seq, dist=2)
            sum_k += (dtw_distance * feature_weight)
        sum_n.append(sum_k)

    tot = 0
    for s in sum_n:
        tot+=s
    return tot/len(sum_n)

def train_weights(w, weights, u, test, train1, train2, train3):
    m = -1
    mis = 0
    print("Feature", w, "Test User", u)

    weights[w] = 0
    r1_1 = dtw(train1, test, weights)
    r2_1 = dtw(train2, test, weights)
    r3_1 = dtw(train3, test, weights)

    weights[w] = 1
    r1_2 = dtw(train1, test, weights)
    r2_2 = dtw(train2, test, weights)
    r3_2 = dtw(train3, test, weights)

    dif1 = r1_2 - r1_1
    dif2 = r2_2 - r2_1
    dif3 = r3_2 - r3_1

    if u == 1:
        return min(dif2, dif3)/dif1
    elif u == 2:
        return min(dif1, dif3) / dif2
    else:
        return min(dif1, dif2) / dif3


    #print("U1, w0: {n1}, w1: {n2}, Dif: {d}".format(n1=r1_1, n2=r1_2, d=r1_2-r1_1))
    #print("U2, w0: {n1}, w1: {n2}, Dif: {d}".format(n1=r2_1, n2=r2_2, d=r2_2-r2_1))
    #print("U3, w0: {n1}, w1: {n2}, Dif: {d}".format(n1=r3_1, n2=r3_2, d=r3_2-r3_1))
    #print()


U1 = extract(dir1a, dir1b, flatten=False)
U2 = extract(dir2a, dir2b, flatten=False)
U3 = extract(dir3a, dir3b, flatten=False)

U1_train = U1[:-1]
U1_test = U1[-1]

U2_train = U2[:-1]
U2_test = U2[-1]

U3_train = U3[:-1]
U3_test = U3[-1]

ws = [0.09, 0.08, 0.06, 0.06, 0.04, 0.06, 0.05, 0.05, 0.07, 0.06, 0.07, 0.05, 0.06, 0.05, 0.06, 0.05, 0.04]

print("User 1:", dtw(U1_train, U3_test, ws))
print("User 2:", dtw(U2_train, U3_test, ws))
print("User 3:", dtw(U3_train, U3_test, ws))

#indices = range(17)
#avr_dif = []
#for w in indices:

 #   ratio1 = train_weights(w, ws, 1, U1_test, U1_train, U2_train, U3_train)
  #  ratio2 = train_weights(w, ws, 2, U2_test, U1_train, U2_train, U3_train)
   # ratio3 = train_weights(w, ws, 3, U3_test, U1_train, U2_train, U3_train)

    #avr_dif.append((ratio1+ratio2+ratio3)/3)
    #print((ratio1+ratio2+ratio3)/3)

#print(avr_dif)
