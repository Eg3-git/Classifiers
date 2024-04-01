import time

import user_model, task_model
from tqdm import tqdm

users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
methods = ["dt"]
# methods = ["svm", "rf", "knn", "dt"]
tasks = ["abc", "cir", "star", "www", "xyz"]


def test_svm():
    es = [1000, 500, 100, 50, 10]

    accs = []
    train_times = []
    pred_times = []

    for e in tqdm(es):

        task_model.train("rf", 1, verbose=False, estimators=e)

        user_acc_total = 0
        total_train_time = 0
        total_pred_time = 0
        for u in users:

            test_data, test_classes, task_classes, train_time = user_model.train(["rf"], u, tasks, 1,
                                                                                    verbose=False)

            user_accuracy, task_accuracy, pred_time = user_model.test(
                            u, "rf",
                            test_data,
                            test_classes,
                            task_classes,
                            haptics_or_ur3e=1, verbose=False, metrics=False)

            user_acc_total += user_accuracy
            total_train_time += train_time
            total_pred_time += pred_time

        accs.append(user_acc_total / len(users))
        train_times.append(total_train_time / len(users))
        pred_times.append(total_pred_time / len(users))

    print(accs, train_times, pred_times)


test_svm()
