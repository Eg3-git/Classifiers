import csv

from scipy.io import loadmat

import numpy as np
import os.path
import math


def extract(user, task, haptics_or_ur3e=0, interval=100, extract_features=True, split_train_test=True):
    base_dir = "Data Collection/{u}/".format(u=user, t=task)
    features = []

    if haptics_or_ur3e:
        dir_end = "_ur3e_end_effectors_pose.csv"
    else:
        dir_end = "_haptics_end_effector_pose.csv"

    start_seq = f"{user}{task}"
    csv_files = [item for item in os.listdir(base_dir) if item.startswith(start_seq) and item.endswith(dir_end)]

    for instance in csv_files:
        f_dir = base_dir + instance

        with open(f_dir, newline='') as f:
            data = list(csv.reader(f))

            if extract_features:
                features.extend(feature_calc(interval, data))

            else:
                features.extend([[float(j) for j in i] for i in data[1:]])

    if split_train_test:
        l = len(features)
        train = features[:int(l * 0.75)]
        test = features[int(l * 0.75):]

        return train, test
    else:
        return features


def curvature_numerator(velocity, accel):
    c_zy = (accel[2] * velocity[1]) - (accel[1] * velocity[2])
    c_xz = (accel[0] * velocity[2]) - (accel[2] * velocity[0])
    c_yx = (accel[1] * velocity[0]) - (accel[0] * velocity[1])
    return math.sqrt((c_zy ** 2) + (c_xz ** 2) + (c_yx ** 2))


def feature_calc(interval, data):
    t = interval * 0.001
    prev_vel = [0, 0, 0]
    prev_accel = [0, 0, 0]
    features = []
    for i in range(1 + interval, len(data), interval):
        prev_pos = [float(x) for x in data[i - interval][1:4]]
        pos = [float(x) for x in data[i][1:4]]

        s = [pos[0] - prev_pos[0], pos[1] - prev_pos[1], pos[2] - prev_pos[2]]
        pos_diff = math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
        velocity = [s[0] / t, s[1] / t, s[2] / t]
        accel = [(velocity[0] - prev_vel[0]) / t, (velocity[1] - prev_vel[1]) / t,
                 (velocity[2] - prev_vel[2]) / t]
        accel_norm = math.sqrt(accel[0] ** 2 + accel[1] ** 2 + accel[2] ** 2)
        jerk = [(accel[0] - prev_accel[0]) / t, (accel[1] - prev_accel[1]) / t,
                (accel[2] - prev_accel[2]) / t]
        try:
            slope_xy = np.arctan(velocity[1] / velocity[0])
            slope_zx = np.arctan(velocity[0] / velocity[2])
            curvature = curvature_numerator(velocity, accel) / (
                    (velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2) ** (3 / 2))
        except ZeroDivisionError:
            slope_xy = 0
            slope_zx = 0
            curvature = 0
        to_append = [pos[0], pos[1], pos[2], pos_diff, velocity[0], velocity[1], velocity[2], accel[0],
                     accel[1], accel[2], accel_norm, jerk[0], jerk[1], jerk[2], slope_xy, slope_zx,
                     curvature]
        features.append(to_append)

        prev_vel = velocity
        prev_accel = accel
    return features
# out = extract(dir1a, dir1b)
# print(out)
