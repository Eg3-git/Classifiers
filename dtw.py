import csv

from feature_extraction import extract
from dtaidistance import dtw
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

tasks = ["abc", "cir", "star", "www", "xyz"]
users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]


def calc_dtw(s1, s2, weights):
    s1 = np.array(s1)
    s2 = np.array(s2)
    weighted_distance = 0
    for feature in range(17):
        feature_series_1 = s1[:, feature]
        feature_series_2 = s2[:, feature]
        weighted_distance += (dtw.distance(feature_series_1, feature_series_2) * weights[feature])

    return weighted_distance


def weight_train_task(sample_ratio=0.1):
    to_be_minimised = {}
    to_be_maximised = {}
    all_task_data = {}
    for task in tasks:
        all_task_data[task] = []
        for user in users:
            train, _ = extract(user, task, haptics_or_ur3e=1, interval=10)
            all_task_data[task].extend(train)

        to_be_minimised[task] = np.zeros((17,))
        to_be_maximised[task] = np.zeros((17,))

    for task1 in tqdm(tasks):
        for i in range(0, int(len(all_task_data[task1]) * sample_ratio), 10):
            for task2 in tasks:
                for j in range(0, int(len(all_task_data[task2]) * sample_ratio), 10):
                    for f in range(17):
                        full_weight = np.zeros((17,))
                        full_weight[f] = 1

                        dist = calc_dtw(all_task_data[task1][i:i + 10], all_task_data[task2][j:j + 10],
                                        full_weight)

                        if task1 == task2:
                            to_be_minimised[task1][f] += dist
                        else:
                            to_be_maximised[task1][f] += dist

    with open("task_weight_results.csv", "w") as results:
        writer = csv.writer(results)
        for task in tasks:
            row = []
            ratio_max_to_min = to_be_maximised[task] / to_be_minimised[task]
            row.append(task)
            r_total = ratio_max_to_min.sum()
            for f in range(17):
                row.append(ratio_max_to_min[f] / r_total)

            writer.writerow(row)


def load_task_weights():
    task_weights = {}
    with open("weight_results.csv", "r") as input_file:
        raw_data = csv.reader(input_file)

        for line in raw_data:
            task_weights[line[0]] = np.zeros((17,))
            for f in range(17):
                task_weights[line[0]][f] = float(line[f + 1])

    return task_weights


def task_train(n=10, sample_ratio=0.1):
    task_weights = load_task_weights()
    best_samples = {}
    test_data = {}
    for task in tqdm(tasks):
        best_samples[task] = []
        task_data = []
        test_data[task] = []
        dtw_map = {}
        for user in users:
            train, test = extract(user, task, haptics_or_ur3e=1, interval=10)
            task_data.extend(train)
            test_data[task].extend(test)

        max_iter = int(len(task_data) * sample_ratio) - 10
        for i in range(0, max_iter, 10):
            total_distance = 0
            for j in range(0, max_iter, 10):
                total_distance += calc_dtw(task_data[i:i + 10], task_data[j:j + 10], task_weights[task])
            dtw_map[i] = total_distance

        best_indices = {k: v for k, v in sorted(dtw_map.items(), key=lambda item: item[1])[:n]}
        for k in best_indices.keys():
            best_samples[task].extend(task_data[k:k + 10])

    print(best_samples)
    with open("dtw_task_model_ur3e.csv", "w") as model:
        writer = csv.writer(model)
        for task, samples in best_samples.items():
            row = [task] + [x for point in samples for x in point]
            writer.writerow(row)

    return test_data


def load_task_model():
    task_data = {}
    with open("dtw_task_model_ur3e.csv", "r") as model:
        raw_data = csv.reader(model)
        for line in raw_data:
            task_data[line[0]] = []
            for p in range(1, len(line), 17):
                point = []
                for x in range(17):
                    point.append(float(line[p + x]))
                task_data[line[0]].append(point)

    return task_data


def weight_train_user(sample_ratio=0.1):
    for task in tqdm(tasks):
        user_weights = {u: np.zeros((17,)) for u in users}
        to_be_minimised = {}
        to_be_maximised = {}
        task_user_data = {}

        for user in users:
            train, _ = extract(user, task, haptics_or_ur3e=1, interval=10)
            task_user_data[user] = train

            to_be_minimised[user] = np.zeros((17,))
            to_be_maximised[user] = np.zeros((17,))

        for user1 in users:
            for i in range(0, int(len(task_user_data[user1]) * sample_ratio), 10):
                for user2 in users:
                    for j in range(0, int(len(task_user_data[user2]) * sample_ratio), 10):
                        for f in range(17):
                            full_weight = np.zeros((17,))
                            full_weight[f] = 1

                            dist = calc_dtw(task_user_data[user1][i:i + 10], task_user_data[user2][j:j + 10],
                                            full_weight)

                            if user1 == user2:
                                to_be_minimised[user1][f] += dist
                            else:
                                to_be_maximised[user1][f] += dist

            ratio_max_to_min = to_be_maximised[user1] / to_be_minimised[user1]
            r_total = ratio_max_to_min.sum()
            user_weights[user1] = (ratio_max_to_min / r_total)

        with open(f"models/dtw/{task}/{task}_user_weight_results.csv", "w") as results:
            writer = csv.writer(results)
            for user in users:
                row = [user]
                for f in range(17):
                    row.append(user_weights[user][f])

                writer.writerow(row)


def user_train(task, sample_ratio=0.1, n=10):
    user_weights = load_user_weights(task)
    test_data = {}
    for user in tqdm(users):
        best_samples = []

        train, test = extract(user, task, haptics_or_ur3e=1, interval=10)
        test_data[user] = test

        max_iter = int(len(train) * sample_ratio) - 10
        dtw_map = {}
        for i in range(0, max_iter, 10):
            total_distance = 0
            for j in range(0, max_iter, 10):
                total_distance += calc_dtw(train[i:i + 10], train[j:j + 10], user_weights[user])

            dtw_map[i] = total_distance

        best_indices = {k: v for k, v in sorted(dtw_map.items(), key=lambda item: item[1])[:n]}
        for k in best_indices.keys():
            best_samples.extend(train[k:k + 10])

        with open(f"models/dtw/{task}/dtw_{user}_{task}_ur3e.csv", "w") as model:
            writer = csv.writer(model)
            for vector in best_samples:
                writer.writerow(vector)

    return test_data


def big_test(sample_ratio=0.1):
    task_model = load_task_model()
    task_weights = load_task_weights()
    all_test_data = {}
    correct_task = 0
    correct_user = 0
    total = 0
    for task in tasks:
        all_test_data[task] = user_train(task)

    for task in tqdm(tasks):
        for user in users:
            for i in range(0, int(len(all_test_data[task][user])*sample_ratio), 10):

                min_task_distance = -1
                best_task = ""
                for t in tasks:
                    for j in range(0, len(task_model[t]), 10):
                        distance = calc_dtw(all_test_data[task][user][i:i + 10], task_model[t][j:j + 10],
                                            task_weights[t])
                        if distance < min_task_distance or min_task_distance == -1:
                            min_task_distance = distance
                            best_task = t

                user_weights = load_user_weights(best_task)
                min_user_distance = -1
                user_pred = ""
                for u in users:
                    user_model = load_user_model(u, best_task)
                    for j in range(0, len(user_model), 10):
                        distance = calc_dtw(all_test_data[task][user][i:i + 10], user_model[j:j + 10], user_weights[u])
                        if distance < min_user_distance or min_user_distance == -1:
                            min_user_distance = distance
                            user_pred = u

                if best_task == task:
                    correct_task += 1
                if user_pred == user:
                    correct_user += 1

                total += 1
    print("Task accuracy:", correct_task / total)
    print("User accuracy:", correct_user / total)


def load_user_model(user, task):
    user_data = []
    with open(f"models/dtw/{task}/dtw_{user}_{task}_ur3e.csv", "r") as model:
        raw_data = csv.reader(model)
        for line in raw_data:
            point = []
            for val in line:
                point.append(float(val))
            user_data.append(point)

    return user_data


def load_user_weights(task):
    user_weights = {}
    with open(f"models/dtw/{task}/{task}_user_weight_results.csv", "r") as input_file:
        raw_data = csv.reader(input_file)

        for line in raw_data:
            user_weights[line[0]] = np.zeros((17,))
            for f in range(17):
                user_weights[line[0]][f] = float(line[f + 1])

    return user_weights


# weight_train(0.1)
# weight_train_user(0.1)
# d = task_train()
# task_test(d)

big_test()
