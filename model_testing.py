import user_model, task_model
from tqdm import tqdm

users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
methods = ["dt"]
# methods = ["svm", "rf", "knn", "dt"]
tasks = ["abc", "cir", "star", "www", "xyz"]


def test_svm():
    kernels = ['sigmoid', 'rbf']
    gamma = ["scale", "auto"]

    best_para_task = {"kernel": "", "gamma": ""}
    best_para_user = {"kernel": "", "gamma": ""}
    best_acc = 0

    for k in tqdm(kernels):
        for g in gamma:
            task_paras = {"kernel": k, "gamma": g}
            task_model.train("svm", 1, verbose=False, paras=task_paras)

            for k2 in kernels:

                for g2 in gamma:
                    user_acc_total = 0
                    for u in users:
                        user_paras = {"kernel": k2, "gamma": g2}
                        test_data, test_classes, task_classes, _ = user_model.train(["svm"], u, tasks, 1,
                                                                                    verbose=False, paras=user_paras)

                        user_accuracy, task_accuracy, _ = user_model.test(
                            u, "svm",
                            test_data,
                            test_classes,
                            task_classes,
                            haptics_or_ur3e=1, verbose=False, metrics=False)

                        user_acc_total += user_accuracy

                    if user_acc_total / len(users) > best_acc:
                        best_acc = user_acc_total / len(users)
                        best_para_task["kernel"] = k

                        best_para_task["gamma"] = g

                        best_para_user["kernel"] = k2

                        best_para_user["gamma"] = g2

    print(best_acc, best_para_task, best_para_user)


test_svm()
