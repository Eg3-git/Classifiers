import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import task_model
import user_model

users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
# methods = ["dt"]
methods = ["svm", "rf", "knn", "dt"]
tasks = ["abc", "cir", "star", "www", "xyz"]


def bulk_test():
    results = [{}, {}]

    for hou in range(2):

        for m in methods:
            results[hou][m] = {}
            results[hou][m]["Task Model Train Time"] = task_model.train(m, hou)

            train_time_total = 0
            user_acc_total = 0
            task_acc_total = 0
            user_lls_total = 0
            task_lls_total = 0
            user_auc_total = 0
            pred_time_total = 0
            user_f1_total = 0
            task_f1_total = 0
            task_confusion_matrix = np.zeros((5, 5))
            user_confusion_matrix = np.zeros((2, 2))
            for u in users:
                test_data, test_classes, task_classes, train_time_one_user = user_model.train([m], u, tasks, hou)
                train_time_total += train_time_one_user

                user_accuracy, task_accuracy, avr_pred_time, user_lls, task_lls, ucm, tcm, u_auc, t_f1, u_f1 = user_model.test(
                    u, m,
                    test_data,
                    test_classes,
                    task_classes,
                    haptics_or_ur3e=hou, verbose=False)

                user_acc_total += user_accuracy
                task_acc_total += task_accuracy
                pred_time_total += avr_pred_time
                user_lls_total += user_lls
                task_lls_total += task_lls
                user_confusion_matrix += ucm
                task_confusion_matrix += tcm
                user_auc_total += u_auc
                user_f1_total += u_f1
                task_f1_total += t_f1

            results[hou][m]["User Models Average Train Time"] = train_time_total / len(users)
            results[hou][m]["User Prediction Accuracy"] = user_acc_total / len(users)
            results[hou][m]["Task Prediction Accuracy"] = task_acc_total / len(users)
            results[hou][m]["Average Prediction Time"] = pred_time_total / len(users)
            results[hou][m]["Average User Log Loss Score"] = user_lls_total / len(users)
            results[hou][m]["Average Task Log Loss Score"] = task_lls_total / len(users)
            results[hou][m]["User Confusion Matrix Score"] = np.sum(np.trace(user_confusion_matrix)) / np.sum(
                user_confusion_matrix)
            results[hou][m]["Task Confusion Matrix Score"] = np.sum(np.trace(task_confusion_matrix)) / np.sum(
                task_confusion_matrix)
            results[hou][m]["User AUC Score"] = user_auc_total / len(users)
            results[hou][m]["User F1 Score"] = user_f1_total / len(users)
            results[hou][m]["Task F1 Score"] = task_f1_total / len(users)

    with open("results.txt", "w") as f:
        for hou in range(2):
            if hou == 0:
                f.write("HAPTICS\n")
            else:
                f.write("UR3E\n")

            for m, data in results[hou].items():
                f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                f.write(f"{m}\n")
                for metric, value in data.items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n")

            f.write("=======================================================\n")


def find_best_interval():
    intervals = [50, 100, 150, 200, 250, 300]
    for hou in range(1, 2):
        all_user_accuracies = {m: [] for m in methods}
        all_task_accuracies = {m: [] for m in methods}

        for interval in tqdm(intervals):

            for m in methods:

                _ = task_model.train(m, hou, interval, verbose=False)
                avg_user_accuracy = 0
                avg_task_accuracy = 0
                for u in users:
                    test_data, test_classes, task_classes, _ = user_model.train([m], u, tasks, hou, interval,
                                                                                verbose=False)

                    user_accuracy, task_accuracy, _, _, _ = user_model.test(u, m, test_data, test_classes, task_classes,
                                                                            hou, verbose=False)
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
                f"{m} The highest user accuracy was found at: {max(range(len(all_user_accuracies[m])), key=all_user_accuracies[m].__getitem__) + 10}")
            print(
                f"{m} The highest task accuracy was found at: {max(range(len(all_task_accuracies[m])), key=all_task_accuracies[m].__getitem__) + 10}")


bulk_test()
# find_best_interval()
