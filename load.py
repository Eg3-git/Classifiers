import random
import csv
import os

from scipy.io import loadmat
import joblib
from sklearn import svm
import numpy as np
import math
from feature_extraction import extract

dir1a = "3USER_10TASKS/index/CIRCLE1/U3_circle.csv"
dir1b = "3USER_10TASKS/robot-endeffector/CIRCLE1/U3_circle-pos3.mat"

model = joblib.load("ABC_model_rf.joblib")

data = extract(dir1a, dir1b)
X = []
for _ in range(20):
    i = random.randint(0, len(data)-1)
    X.append(data[i])

# result = model.decision_function(X)
result2 = model.predict(X)
print(result2)
