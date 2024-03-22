import csv

from feature_extraction import extract
from dtaidistance import dtw
import numpy as np
from tqdm import tqdm

# ws = [0.09, 0.08, 0.06, 0.06, 0.04, 0.06, 0.05, 0.05, 0.07, 0.06, 0.07, 0.05, 0.06, 0.05, 0.06, 0.05, 0.04]
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


def weight_train(sample_ratio=0.1):
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

    with open("weight_results.csv", "w") as results:
        writer = csv.writer(results)
        for task in tasks:
            row = []
            ratio_max_to_min = to_be_maximised[task] / to_be_minimised[task]
            row.append(task)
            r_total = ratio_max_to_min.sum()
            for f in range(17):
                row.append(ratio_max_to_min[f] / r_total)

            writer.writerow(row)


def load_weights():
    task_weights = {}
    with open("weight_results.csv", "r") as input_file:
        raw_data = csv.reader(input_file)

        for line in raw_data:
            task_weights[line[0]] = np.zeros((17,))
            for f in range(17):
                task_weights[line[0]][f] = float(line[f + 1])

    return task_weights


def task_train(n=10, sample_ratio=0.1):
    task_weights = load_weights()
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

    with open("dtw_task_model_ur3e.csv", "w") as model:
        writer = csv.writer(model)
        for task, samples in best_samples.items():
            row = [task] + [x for point in samples for x in point]
            writer.writerow(row)

    print(len(best_samples["abc"]))
    return test_data


def task_test(test_data):
    task_weights = load_weights()
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

    score = 0
    total = 0
    for actual in tqdm(tasks):
        for i in range(0, len(test_data[actual]), 10):
            task_distances = {task: 0 for task in tasks}
            for task in tasks:
                for j in range(0, len(task_data[task]), 10):
                    task_distances[task] += calc_dtw(test_data[actual][i:i + 10], task_data[task][j:j + 10],
                                                     task_weights[task])
            prediction = min(task_distances, key=task_distances.get)
            if prediction == actual:
                score += 1
            total += 1

    print(score / total)


def user_train():
    for user in users:
        user_data = []
        dtw_map = {}
        for task in tasks:
            train, test = extract(user, task, haptics_or_ur3e=1, interval=10)
            user_data.extend(train)

        print("Calculating distance for", user)
        for i in tqdm(range(0, len(user_data) - 10, 10)):
            total_distance = 0
            for j in range(0, len(user_data) - 10, 10):
                total_distance += calc_dtw(user_data[i:i + 10], user_data[j:j + 10])
            dtw_map[i] = total_distance

        print({k: v for k, v in sorted(dtw_map.items(), key=lambda item: item[1])})


# weight_train(0.1)
d = task_train()
task_test(d)
