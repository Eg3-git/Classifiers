from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score, f1_score
from feature_extraction import extract
from joblib import dump, load
import time
import numpy as np

intervals = [100]


def train(methods, train_user, tasks_to_train, haptics_or_ur3e=0, interval=100, verbose=True, paras={}):
    all_users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]

    test_data = []
    test_classes = []
    task_classes = []
    t = 0
    tot_time = 0
    for task in tasks_to_train:
        train_data = []
        train_classes = []
        for u in all_users:
            train, test = extract(u, task, haptics_or_ur3e, interval=interval)
            train_data.extend(train)
            train_classes.extend([0 for _ in train] if u == train_user else [1 for _ in train])
            test_data.extend(test)
            test_classes.extend([0 for _ in test] if u == train_user else [1 for _ in test])
            task_classes.extend([t for _ in test])
        t += 1

        for method in methods:
            if verbose:
                print("Training {m} model for {u} {t} with {n} data points".format(m=method, u=train_user, t=task,
                                                                                   n=len(train_data)))

            if method == "svm":
                model = SVC(probability=True, kernel=paras["kernel"], gamma=paras["gamma"])
            elif method == "rf":
                model = RandomForestClassifier()
            elif method == "knn":
                model = KNeighborsClassifier()
            elif method == "dt":
                model = DecisionTreeClassifier()
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


def test(user, method, test_data, test_classes, task_classes, haptics_or_ur3e=0, verbose=True, metrics=True):
    tasks = ["abc", "cir", "star", "www", "xyz"]
    name = "ur3e" if haptics_or_ur3e else "haptics"
    task_model = load("{m}_task_model_{h}.joblib".format(m=method, h=name))
    user_predictions = []
    task_predictions = []
    user_probs = []

    tot_time = 0

    for i in range(len(test_data)):
        t1 = time.time()
        predicted_task = task_model.predict([test_data[i]])[0]

        user_model = load(
            "models/{m}/{t}/{m}_{u}_{t}_{h}.joblib".format(m=method, t=tasks[predicted_task], u=user, h=name))

        current_prediction = user_model.predict([test_data[i]])[0]

        t2 = time.time()
        tot_time += (t2 - t1)

        task_predictions.append(predicted_task)
        user_predictions.append(current_prediction)
        if metrics:
            user_probs.append(user_model.predict_proba(test_data))

    user_accuracy = accuracy_score(user_predictions, test_classes)
    task_accuracy = accuracy_score(task_predictions, task_classes)
    avr_pred_time = tot_time / len(test_data)

    if metrics:
        task_probs = task_model.predict_proba(test_data)
        user_probs_avg = np.mean(user_probs, axis=0)
        user_log_loss_score = log_loss(test_classes, user_probs_avg)
        task_log_loss_score = log_loss(task_classes, task_probs)
        user_confusion_matrix = confusion_matrix(user_predictions, test_classes)
        task_confusion_matrix = confusion_matrix(task_predictions, task_classes)
        user_auc_score = roc_auc_score(test_classes, user_probs_avg[:, 1])
        task_f1 = f1_score(task_predictions, task_classes, average='weighted')
        user_f1 = f1_score(user_predictions, test_classes)

    if verbose:
        print("Testing points:", len(test_data))
        print("Users correctly classified:", user_accuracy)
        print("Tasks correctly classified:", task_accuracy)
        # print("Maximum number of false breaches:", max_false_negs)
        # print("Maximum number of missed breaches:", max_false_pos)
        print("Avr prediction time:", avr_pred_time)

    if metrics:
        return user_accuracy, task_accuracy, avr_pred_time, user_log_loss_score, task_log_loss_score, user_confusion_matrix, task_confusion_matrix, user_auc_score, task_f1, user_f1
    else:
        return user_accuracy, task_accuracy, avr_pred_time
