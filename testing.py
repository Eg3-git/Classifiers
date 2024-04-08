import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

import task_model
import user_model

users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
intervals = [5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 650, 700, 750, 800, 850, 900, 950]
# intervals = [100, 200, 300, 400, 500]
# methods = ["knn"]
methods = ["svm", "rf", "knn", "dt"]
methods_caps = [s.upper() for s in methods]
tasks = ["abc", "cir", "star", "www", "xyz"]


def bulk_test():
    print("General Metris")
    metric_test()

    print("RF Task")
    metric_test(use_task_model="rf", f_name="rf_task")

    print("DT Task")
    metric_test(use_task_model="dt", f_name="dt_task")


def metric_test(use_task_model=None, f_name="general"):
    results = {m: {i: {} for i in intervals} for m in methods}

    for i in tqdm(intervals):
        if use_task_model is not None:
            task_model.train(use_task_model, haptics_or_ur3e=1,
                             interval=i, verbose=False)

        for m in methods:
            if use_task_model is None:
                task_model.train(m, haptics_or_ur3e=1,
                                 interval=i, verbose=False)

            task_train_time = results[m][i]["Task Model Train Time"] = 0

            train_time_total = 0
            user_acc_total = 0
            task_acc_total = 0
            user_auc_total = 0
            pred_time_total = 0
            pos_f1_total = []
            neg_f1_total = []
            task_confusion_matrix = np.zeros((5, 5))
            user_confusion_matrix = np.zeros((2, 2))
            pos_preds = []
            neg_preds = []
            for u in users:
                test_data, test_classes, task_classes, train_time_one_user = user_model.train([m], u, tasks,
                                                                                              haptics_or_ur3e=1,
                                                                                              interval=i, verbose=False)
                train_time_total += train_time_one_user

                user_accuracy, task_accuracy, avr_pred_time, ucm, tcm, u_auc, p_f1, n_f1, p_preds, n_preds = user_model.test(
                    u, m,
                    test_data,
                    test_classes,
                    task_classes,
                    haptics_or_ur3e=1, verbose=False, metrics=True, true_task=False, use_task_model="dt")

                user_acc_total += user_accuracy
                task_acc_total += task_accuracy
                pred_time_total += avr_pred_time
                user_confusion_matrix += ucm
                task_confusion_matrix += tcm
                user_auc_total += u_auc
                pos_f1_total.extend(p_f1)
                neg_f1_total.extend(n_f1)
                pos_preds.extend(p_preds)
                neg_preds.extend(n_preds)

            results[m][i]["User Confusion Matrix"] = user_confusion_matrix
            results[m][i]["Task Confusion Matrix"] = task_confusion_matrix
            results[m][i]["Task Model Train Time"] = task_train_time
            results[m][i]["User Models Average Train Time"] = train_time_total / len(users)
            results[m][i]["User Prediction Accuracy"] = user_acc_total / len(users)
            results[m][i]["Task Prediction Accuracy"] = task_acc_total / len(users)
            results[m][i]["Average Prediction Time"] = pred_time_total / len(users)
            results[m][i]["User Confusion Matrix Score"] = np.sum(np.trace(user_confusion_matrix)) / np.sum(
                user_confusion_matrix)
            results[m][i]["Task Confusion Matrix Score"] = np.sum(np.trace(task_confusion_matrix)) / np.sum(
                task_confusion_matrix)
            results[m][i]["User AUC Score"] = user_auc_total / len(users)
            results[m][i]["Positive F1 Score"] = f1_score([0 for _ in pos_f1_total], pos_f1_total)
            results[m][i]["Negative F1 Score"] = f1_score([0 for _ in neg_f1_total], neg_f1_total)
            results[m][i]["Positive AUC Score"] = roc_auc_score([1 for _ in pos_preds], pos_preds)
            results[m][i]["Negative AUC Score"] = roc_auc_score([1 for _ in neg_preds], neg_preds)

    plot_line(intervals, [[results[m][i]["Task Model Train Time"] for i in intervals] for m in methods], methods_caps,
              "Time Interval", "Train Time (s)", "Time Interval on Task Model Train Time")
    plot_line(intervals, [[results[m][i]["User Models Average Train Time"] for i in intervals] for m in methods],
              methods_caps,
              "Time Interval", "Train Time (s)", "Time Interval on User Model Train Time")
    plot_line(intervals, [
        [(results[m][i]["User Confusion Matrix"][0, 0] / np.sum(results[m][i]["User Confusion Matrix"][:, 0]) * 100) for
         i in intervals] for m in methods], methods_caps,
              "Time Interval", "Accuracy (%)", "Time Interval on User Prediction Accuracy")
    plot_line(intervals, [
        [(results[m][i]["User Confusion Matrix"][1, 1] / np.sum(results[m][i]["User Confusion Matrix"][:, 1]) * 100) for
         i in
         intervals] for m in methods], methods_caps,
              "Time Interval", "Accuracy (%)", "Time Interval on Attacker Prediction Accuracy")

    plot_line(intervals, [[results[m][i]["Task Prediction Accuracy"] * 100 for i in intervals] for m in methods],
              methods_caps,
              "Time Interval", "Accuracy (%)", "Time Interval on Task Prediction Accuracy")

    plot_line(intervals, [[results[m][i]["Average Prediction Time"] / 1000 for i in intervals] for m in methods],
              methods_caps,
              "Time Interval", "Prediction Time (ms)", "Time Interval on Prediction Time")

    with open(f_name + "_results.txt", "w") as f:
        for i in intervals:
            f.write(f"i={i}\n")
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            for m in methods:
                f.write(f"{m}\n")
                for metric, data in results[m][i].items():
                    f.write(f"{metric}: {data}\n")
                f.write("\n")

            f.write("=======================================================\n")


def plot_line(x, ys, labels, x_label, y_label, title):
    for i in range(len(ys)):
        plt.plot(x, ys[i], label=labels[i])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def find_best_interval():
    for hou in range(1, 2):
        all_user_accuracies = {m: [] for m in methods}
        all_task_accuracies = {m: [] for m in methods}

        for interval in tqdm(intervals):

            for m in methods:

                task_model.train(m, haptics_or_ur3e=hou, interval=interval, verbose=False)
                avg_user_accuracy = 0
                avg_task_accuracy = 0
                for u in users:
                    test_data, test_classes, task_classes, _ = user_model.train([m], u, tasks, hou, interval,
                                                                                verbose=False)

                    user_accuracy, task_accuracy, _ = user_model.test(u, m, test_data, test_classes, task_classes,
                                                                      haptics_or_ur3e=hou, verbose=False, metrics=False)
                    avg_user_accuracy += user_accuracy
                    avg_task_accuracy += task_accuracy
                all_user_accuracies[m].append(avg_user_accuracy / len(users))
                all_task_accuracies[m].append(avg_task_accuracy / len(users))

        for m in methods:
            plt.plot(intervals, all_user_accuracies[m])
            plt.plot(intervals, all_task_accuracies[m])
            plt.title(f'User and task accuracies over interval - {m}')
            plt.xlabel('Interval')
            plt.ylabel('User accuracy')
            plt.show()

            print(
                f"{m} The highest user accuracy was found at: {max(range(len(all_user_accuracies[m])), key=all_user_accuracies[m].__getitem__)}")
            print(
                f"{m} The highest task accuracy was found at: {max(range(len(all_task_accuracies[m])), key=all_task_accuracies[m].__getitem__)}")


def auc(m):
    scores = []
    for i in tqdm(intervals):
        for u in users:
            test_data, test_classes, task_classes, train_time_one_user = user_model.train([m], u, tasks,
                                                                                          haptics_or_ur3e=1,
                                                                                          interval=i, verbose=False)
        scores.append(user_model.calc_auc(m, haptics_or_ur3e=1, interval=i))
    print(scores)


bulk_test()
# auc("rf")
# find_best_interval()
