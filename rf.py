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
import math
from joblib import dump
from feature_extraction import extract

dir1a = "3USER_10TASKS/index/ABC1/U1_abc.csv"
dir1b = "3USER_10TASKS/robot-endeffector/ABC1/U1_ABC-pos3.mat"
#dir1a = "3USER_10TASKS/index/O1/U1_o.csv"
#dir1b = "3USER_10TASKS/robot-endeffector/O1/U1_O-pos3.mat"
#dir1e = "3USER_10TASKS/index/P2P2/U1_p2p2.csv"
#dir1f = "3USER_10TASKS/robot-endeffector/P2P2/U1_p2p2-pos3.mat"

dir2a = "3USER_10TASKS/index/ABC1/U2_abc.csv"
dir2b = "3USER_10TASKS/robot-endeffector/ABC1/U2_ABC-pos3.mat"
#dir2a = "3USER_10TASKS/index/O1/U2_o.csv"
#dir2b = "3USER_10TASKS/robot-endeffector/O1/U2_O-pos3.mat"
#dir2e = "3USER_10TASKS/index/P2P2/U2_p2p2.csv"
#dir2f = "3USER_10TASKS/robot-endeffector/P2P2/U2_p2p2-pos3.mat"

dir3a = "3USER_10TASKS/index/ABC1/U3_abc.csv"
dir3b = "3USER_10TASKS/robot-endeffector/ABC1/U3_ABC-pos3.mat"
#dir3a = "3USER_10TASKS/index/O1/U3_o.csv"
#dir3b = "3USER_10TASKS/robot-endeffector/O1/U3_O-pos3.mat"
#dir3e = "3USER_10TASKS/index/P2P2/U3_p2p2.csv"
#dir3f = "3USER_10TASKS/robot-endeffector/P2P2/U3_p2p2-pos3.mat"

intervals = [100]
X = []
Y = []
Z = []

for t in intervals:
    Xs = extract(dir1a, dir1b, t)
    Ys = extract(dir2a, dir2b, t)
    Zs = extract(dir3a, dir3b, t)

    for d1 in Xs:
        for d2 in d1:
            X.append(d2)

    for d1 in Ys:
        for d2 in d1:
            Y.append(d2)

    for d1 in Zs:
        for d2 in d1:
            Z.append(d2)


    classes = [1 for _ in range(len(X))] + [2 for _ in range(len(Y))] + [3 for _ in range(len(Z))]
    data = X + Y + Z

    data_train, data_test, classes_train, classes_test = train_test_split(data, classes, test_size=0.25, shuffle=True)


    print("started", len(data))
    model = RandomForestClassifier()
    print("training t = ", t)
    model.fit(data_train, classes_train)
    print("Accuracy")
    predictions = model.predict(data_test)
    print(accuracy_score(predictions, classes_test))
print("saving")
dump(model, "ABC_model_rf.joblib")
print("done")
