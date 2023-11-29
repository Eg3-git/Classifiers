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

def distance(x, y):
    distances = np.zeros((len(x), len(y)))
    for i in range(len(y)):
        for j in range(len(x)):
            distances[i, j] = (x[j]-y[i])**2
    return distances

def calc_cost(x, y):
    distances = distance(x, y)

    cost = np.zeros(len(x), len(y))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]

    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]

    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1]) + distances[i, j]

    return cost

def dtw(train, test):
    sum_n = []
    for n in train:
        sum_k = 0
        for k in range(17):
            train_seq = [o[k] for o in n]
            test_seq = [o[k] for o in test]

            dtw_distance, warp_path = fastdtw(test_seq, train_seq, dist=2)
            sum_k += dtw_distance
        sum_n.append(sum_k)

    tot = 0
    for s in sum_n:
        tot+=s
    return tot/len(sum_n)

U1 = extract(dir1a, dir1b)
U2 = extract(dir2a, dir2b)
U3 = extract(dir3a, dir3b)

U1_train = U1[:-1]
U1_test = U1[-1]

U2_train = U2[:-1]
U2_test = U2[-1]

U3_train = U3[:-1]
U3_test = U3[-1]

print("User 1:", dtw(U1_train, U3_test))
print("User 2:", dtw(U2_train, U3_test))
print("User 3:", dtw(U3_train, U3_test))


