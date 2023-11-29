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

# dir1a = "3USER_10TASKS/index/CIRCLE1/U1_circle.csv"
# dir1b = "3USER_10TASKS/robot-endeffector/CIRCLE1/U1_circle-pos3.mat"

dir1a = "3USER_10TASKS/index/ABC1/U1_abc.csv"
dir1b = "3USER_10TASKS/robot-endeffector/ABC1/U1_ABC-pos3.mat"


def extract(dir1, dir2, interval=100):
    with open(dir1, newline='') as f:
        X = []
        indices = csv.reader(f)
        mat = loadmat(dir2)['pos3']
        t = interval * 0.001

        for i, j in indices:
            Y = []
            a = int(i)
            b = int(i) + interval
            prev_vel = [0, 0, 0]
            prev_accel = [0, 0, 0]

            while b < int(j):
                prev_pos = mat[a]
                pos = mat[b]
                s = [pos[0] - prev_pos[0], pos[1] - prev_pos[1], pos[2] - prev_pos[2]]
                pos_diff = math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
                velocity = [s[0] / t, s[1] / t, s[2] / t]
                accel = [(velocity[0] - prev_vel[0]) / t, (velocity[1] - prev_vel[1]) / t,
                         (velocity[2] - prev_vel[2]) / t]
                accel_norm = math.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
                jerk = [(accel[0] - prev_accel[0]) / t, (accel[1] - prev_accel[1]) / t, (accel[2] - prev_accel[2]) / t]
                slope_xy = np.arctan(velocity[1] / velocity[0])
                slope_zx = np.arctan(velocity[0] / velocity[2])
                curvature = curvature_numerator(velocity, accel) / ((velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**(3/2))

                to_append = [pos[0], pos[1], pos[2], pos_diff, velocity[0], velocity[1], velocity[2], accel[0], accel[1], accel[2], accel_norm, jerk[0], jerk[1], jerk[2], slope_xy, slope_zx, curvature]
                Y.append(to_append)

                prev_vel = velocity
                prev_accel = accel
                a = b
                b += interval

            X.append(Y)

        return X


def curvature_numerator(velocity, accel):
    c_zy = (accel[2] * velocity[1]) - (accel[1] * velocity[2])
    c_xz = (accel[0] * velocity[2]) - (accel[2] * velocity[0])
    c_yx = (accel[1] * velocity[0]) - (accel[0] * velocity[1])
    return math.sqrt((c_zy ** 2) + (c_xz ** 2) + (c_yx ** 2))
# out = extract(dir1a, dir1b)
# print(out)
