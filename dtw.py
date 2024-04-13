import csv

from feature_extraction import extract
from dtaidistance import dtw
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

tasks = ["abc", "cir", "star", "www", "xyz"]
users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10", "u11"]



def weight_train_task(intervals):
    to_be_minimised = {}
    to_be_maximised = {}

    for inter in tqdm(intervals):
        to_be_minimised[inter] = np.zeros((17,))
        to_be_maximised[inter] = np.zeros((17,))
        all_task_data = {}
        for task in tasks:
            all_task_data[task] = []
            for user in users:
                train, _ = extract(user, task, haptics_or_ur3e=1, interval=inter)
                all_task_data[task].extend(train)


        for task1 in tasks:
            for i in all_task_data[task1]:
                for task2 in tasks:
                    for j in all_task_data[task2]:
                        for f in range(17):
                            dist = abs(i[f] - j[f])

                            if task1 == task2:
                                to_be_minimised[inter][f] += dist
                            else:
                                to_be_maximised[inter][f] += dist

    with open("task_weight_results.csv", "w") as results:
        writer = csv.writer(results)
        for inter in intervals:
            row = []
            ratio_max_to_min = to_be_maximised[inter] / to_be_minimised[inter]
            row.append(inter)
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







def weight_train_user(intervals):

    for task in tasks:
        user_weights = {}
        for inter in tqdm(intervals):
            user_weights[inter] = {u: np.zeros((17,)) for u in users}
            to_be_minimised = {}
            to_be_maximised = {}
            task_user_data = {}

            for user in users:
                train, _ = extract(user, task, haptics_or_ur3e=1, interval=inter)
                task_user_data[user] = train

                to_be_minimised[user] = np.zeros((17,))
                to_be_maximised[user] = np.zeros((17,))

            for user1 in users:
                for i in task_user_data[user1]:
                    for user2 in users:
                        for j in task_user_data[user2]:
                            for f in range(17):

                                dist = abs(i[f] - j[f])

                                if user1 == user2:
                                    to_be_minimised[user1][f] += dist
                                else:
                                    to_be_maximised[user1][f] += dist

                ratio_max_to_min = to_be_maximised[user1] / to_be_minimised[user1]
                r_total = ratio_max_to_min.sum()
                user_weights[inter][user1] = (ratio_max_to_min / r_total)

        for user in users:
            with open(f"models/weights/{task}/{task}_{user}_weight_results.csv", "w") as results:
                writer = csv.writer(results)
                for i in intervals:
                    row = [i]
                    for f in range(17):
                        row.append(user_weights[i][user][f])

                    writer.writerow(row)




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
intervals = [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 650, 700, 750, 800, 850, 900, 950, 1000]
weight_train_user(intervals)
