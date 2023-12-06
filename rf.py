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
# dir1a = "3USER_10TASKS/index/O1/U1_o.csv"
# dir1b = "3USER_10TASKS/robot-endeffector/O1/U1_O-pos3.mat"
# dir1e = "3USER_10TASKS/index/P2P2/U1_p2p2.csv"
# dir1f = "3USER_10TASKS/robot-endeffector/P2P2/U1_p2p2-pos3.mat"

dir2a = "3USER_10TASKS/index/ABC1/U2_abc.csv"
dir2b = "3USER_10TASKS/robot-endeffector/ABC1/U2_ABC-pos3.mat"
# dir2a = "3USER_10TASKS/index/O1/U2_o.csv"
# dir2b = "3USER_10TASKS/robot-endeffector/O1/U2_O-pos3.mat"
# dir2e = "3USER_10TASKS/index/P2P2/U2_p2p2.csv"
# dir2f = "3USER_10TASKS/robot-endeffector/P2P2/U2_p2p2-pos3.mat"

dir3a = "3USER_10TASKS/index/ABC1/U3_abc.csv"
dir3b = "3USER_10TASKS/robot-endeffector/ABC1/U3_ABC-pos3.mat"
# dir3a = "3USER_10TASKS/index/O1/U3_o.csv"
# dir3b = "3USER_10TASKS/robot-endeffector/O1/U3_O-pos3.mat"
# dir3e = "3USER_10TASKS/index/P2P2/U3_p2p2.csv"
# dir3f = "3USER_10TASKS/robot-endeffector/P2P2/U3_p2p2-pos3.mat"

intervals = [100]

for t in intervals:
    X_train, X_test = extract(dir1a, dir1b, t)
    Y_train, Y_test = extract(dir2a, dir2b, t)
    Z_train, Z_test = extract(dir3a, dir3b, t)
    train_data = X_train + Y_train + Z_train
    train_classes = [1 for _ in X_train] + [2 for _ in Y_train] + [3 for _ in Z_train]
    test_data = X_test + Y_test + Z_test
    test_classes = [1 for _ in X_test] + [2 for _ in Y_test] + [3 for _ in Z_test]



    # data_train, data_test, classes_train, classes_test = train_test_split(data, classes, test_size=0.25, shuffle=True)

    print("started", len(train_data))
    model = RandomForestClassifier()
    print("training t = ", t)
    model.fit(train_data, train_classes)
    print("Accuracy")
    predictions = model.predict(test_data)
    print(accuracy_score(predictions, test_classes))
print("saving")
dump(model, "ABC_model_rf.joblib")
print("done")
