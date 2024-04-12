from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score, f1_score
from tqdm import tqdm

from feature_extraction import extract
from joblib import dump, load
import time
import numpy as np

intervals = [100]


def train(methods, train_user, tasks_to_train, haptics_or_ur3e=0, interval=100, verbose=True, paras=[]):
    all_users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10", "u11"]

    test_data = {t: [] for t in tasks_to_train}
    test_classes = {t: [] for t in tasks_to_train}
    task_classes = {t: [] for t in tasks_to_train}
    t = 0
    tot_time = 0
    for task in tasks_to_train:
        train_data = []
        train_classes = []
        for u in all_users:
            train, test = extract(u, task, haptics_or_ur3e, interval=interval)
            train_data.extend(train)
            train_classes.extend([0 for _ in train] if u == train_user else [1 for _ in train])
            test_data[task].extend(test)
            test_classes[task].extend([0 for _ in test] if u == train_user else [1 for _ in test])
            task_classes[task].extend([t for _ in test])
        t += 1

        for method in methods:
            if verbose:
                print("Training {m} model for {u} {t} with {n} data points".format(m=method, u=train_user, t=task,
                                                                                   n=len(train_data)))

            if method == "svm":
                model = SVC(probability=True, kernel='rbf', gamma='auto')
            elif method == "rf":
                model = RandomForestClassifier(n_estimators=100, criterion="entropy", max_features=None)
            elif method == "knn":
                model = KNeighborsClassifier(weights="uniform", n_neighbors=15, algorithm="auto", p=1)
            elif method == "dt":
                model = DecisionTreeClassifier(criterion="entropy", max_features=None, splitter="best")
            else:
                raise Exception("Method must be 'svm', 'rf', 'knn', or 'dt'")

            t1 = time.time()
            model.fit(train_data, train_classes)
            t2 = time.time()
            tot_time += (t2 - t1)

            name = "ur3e" if haptics_or_ur3e else "haptics"
            dump(model, "models/{m}/{t}/{m}_{u}_{t}_{h}.joblib".format(m=method, t=task, u=train_user, h=name))
    avr_time = tot_time / len(task_classes)
    if verbose:
        print("Avr train time:", avr_time)
    return test_data, test_classes, task_classes, avr_time


def test(user, method, test_data, test_classes, task_classes, haptics_or_ur3e=0, verbose=True, metrics=True,
         true_task=False, use_task_model=None):
    tasks = ["abc", "cir", "star", "www", "xyz"]
    name = "ur3e" if haptics_or_ur3e else "haptics"

    if not true_task:
        if use_task_model is None:
            task_model = load("{m}_task_model_{h}.joblib".format(m=method, h=name))
        else:
            task_model = load("{m}_task_model_{h}.joblib".format(m=use_task_model, h=name))

    user_predictions = []
    task_predictions = []
    real_user = []
    real_task = []
    pos_preds = []
    neg_preds = []
    pos_f1 = []
    neg_f1 = []

    tot_time = 0

    for task in test_data:
        task_counts = np.zeros((5,))

        for i in range(len(test_data[task])):
            t1 = time.time()
            if true_task:
                predicted_task = task_classes[task][i]
            else:
                task_counts[task_model.predict([test_data[task][i]])[0]] += 1
                predicted_task = task_counts.argmax()

            user_model = load(
                "models/{m}/{t}/{m}_{u}_{t}_{h}.joblib".format(m=method, t=tasks[predicted_task], u=user, h=name))

            current_prediction = user_model.predict([test_data[task][i]])[0]

            t2 = time.time()

            if test_classes[task][i] == 0:
                pos_preds.append(user_model.predict_proba([test_data[task][i]])[:, 0])
                pos_f1.append(current_prediction)
            else:
                neg_preds.append(user_model.predict_proba([test_data[task][i]])[:, 0])
                neg_f1.append(current_prediction)

            tot_time += (t2 - t1)

            task_predictions.append(predicted_task)
            user_predictions.append(current_prediction)
            real_user.append(test_classes[task][i])
            real_task.append(task_classes[task][i])

    user_accuracy = accuracy_score(user_predictions, real_user)
    task_accuracy = accuracy_score(task_predictions, real_task)
    avr_pred_time = tot_time / len(test_data)

    if metrics:
        # user_probs_avg = np.mean(user_probs, axis=0)
        user_confusion_matrix = confusion_matrix(real_user, user_predictions)
        task_confusion_matrix = confusion_matrix(real_task, task_predictions)
        user_auc_score = 0  # roc_auc_score(test_classes, user_probs_avg[:, 0])


    if verbose:
        print("Testing points:", len(test_data))
        print("Users correctly classified:", user_accuracy)
        print("Tasks correctly classified:", task_accuracy)
        # print("Maximum number of false breaches:", max_false_negs)
        # print("Maximum number of missed breaches:", max_false_pos)
        print("Avr prediction time:", avr_pred_time)

    if metrics:
        return user_accuracy, task_accuracy, avr_pred_time, user_confusion_matrix, task_confusion_matrix, user_auc_score, pos_f1, neg_f1, pos_preds, neg_preds
    else:
        return user_accuracy, task_accuracy, avr_pred_time


def calc_auc(method, haptics_or_ur3e=1, interval=100, verbose=True, metrics=True):
    all_users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
    tasks = ["abc", "cir", "star", "www", "xyz"]
    auc_score_total = 0

    for user in all_users:
        for task in tasks:
            name = "ur3e" if haptics_or_ur3e else "haptics"
            test_data = []
            test_classes = []

            for u in all_users:
                _, test = extract(u, task, haptics_or_ur3e, interval=interval)
                test_data.extend(test)
                test_classes.extend([0 for _ in test] if u == user else [1 for _ in test])

            user_model = load(
                "models/{m}/{t}/{m}_{u}_{t}_{h}.joblib".format(m=method, t=task, u=user, h=name))

            user_probs = user_model.predict_proba(test_data)
            auc_score_total += roc_auc_score(test_classes, user_probs[:, 0])

    return auc_score_total / (len(all_users) * len(tasks))
