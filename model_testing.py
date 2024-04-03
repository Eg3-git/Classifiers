import time
import numpy as np
import user_model, task_model
from tqdm import tqdm

users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]

tasks = ["abc", "cir", "star", "www", "xyz"]
crit = ["gini", "entropy", "log_loss"]
feat = ["sqrt", "log2", None]

def test_svm():


    accs = []
    train_times = []
    pred_times = []

    for c in tqdm(crit):
        for f in feat:
            task_para = [c, f]

            task_model.train("rf", 1, verbose=False, paras=task_para)

            for c2 in crit:
                for f2 in feat:


                    cm = np.zeros((2,2))

                    user_para = [c2, f2]
                    for u in users:


                        test_data, test_classes, task_classes, train_time = user_model.train(["rf"], u, tasks, 1,
                                                                                    verbose=False, paras=user_para)

                        user_accuracy, task_accuracy, pred_time, user_confusion_matrix, task_confusion_matrix, user_auc_score, task_f1, user_f1= user_model.test(
                            u, "rf",
                            test_data,
                            test_classes,
                            task_classes,
                            haptics_or_ur3e=1, verbose=False, metrics=True)

                        cm += user_confusion_matrix


                    accs = cm[0,0] / len(users)


                    print(accs, c, f, c2, f2)

test_svm()
