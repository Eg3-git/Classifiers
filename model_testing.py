import time
import numpy as np
import user_model, task_model
from tqdm import tqdm

users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]

tasks = ["abc", "cir", "star", "www", "xyz"]
crit = ["gini", "entropy", "log_loss"]
feat = ["sqrt", "log2", None]

ns = [0]#range(5, 151, 10)
intervals = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

def test_svm():


    accs = []
    train_times = []
    pred_times = []

    for i in tqdm(intervals):

            task_model.train("knn", 1, verbose=False, interval=i)

            for n2 in ns:



                    cm = np.zeros((2,2))


                    for u in users:


                        test_data, test_classes, task_classes, train_time = user_model.train(["knn"], u, tasks, 1,
                                                                                    verbose=False, interval=i)

                        user_accuracy, task_accuracy, pred_time, user_confusion_matrix, task_confusion_matrix, user_auc_score, task_f1, user_f1= user_model.test(
                            u, "knn",
                            test_data,
                            test_classes,
                            task_classes,
                            haptics_or_ur3e=1, verbose=False, metrics=True)

                        cm += user_confusion_matrix


                    accs = cm[0,0] / (cm[1,0] + cm[0,0])


                    print(accs, n2)

test_svm()
