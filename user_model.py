from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from feature_extraction import extract
from joblib import dump, load
import time

intervals = [100]


def train(methods, train_user, tasks_to_train, haptics_or_ur3e=0):
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
            train, test = extract(u, task, haptics_or_ur3e)
            train_data.extend(train)
            train_classes.extend([0 for _ in train] if u == train_user else [1 for _ in train])
            test_data.extend(test)
            test_classes.extend([0 for _ in test] if u == train_user else [1 for _ in test])
            task_classes.extend([t for _ in test])
        t += 1

        for method in methods:
            print("Training {m} model for {u} {t} with {n} data points".format(m=method, u=train_user, t=task, n=len(train_data)))

            if method == "svm":
                model = SVC()
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
    print("Avr train time:", avr_time)
    return test_data, test_classes, task_classes, avr_time


def test(user, method, test_data, test_classes, task_classes, haptics_or_ur3e=0):
    tasks = ["abc", "cir", "star", "www", "xyz"]
    name = "ur3e" if haptics_or_ur3e else "haptics"
    task_model = load("{m}_task_model_{h}.joblib".format(m=method, h=name))
    user_predictions = []
    task_predictions = []
    tot_time = 0
    false_negs = 0
    max_false_negs = 0
    false_pos = 0
    max_false_pos = 0

    for i in range(len(test_data)):
        t1 = time.time()
        predicted_task = task_model.predict([test_data[i]])[0]
        task_predictions.append(predicted_task)
        user_model = load(
            "models/{m}/{t}/{m}_{u}_{t}_{h}.joblib".format(m=method, t=tasks[predicted_task], u=user, h=name))
        current_prediction = user_model.predict([test_data[i]])[0]

        if test_classes[i] == 0 and current_prediction != 0:
            false_negs += 1
            if false_negs > max_false_negs:
                max_false_negs = false_negs
        else:
            false_negs = 0

        if test_classes[i] != 0 and current_prediction == 0:
            false_pos += 1
            if false_pos > max_false_pos:
                max_false_pos = false_pos
        else:
            false_pos = 0

        user_predictions.append(current_prediction)
        t2 = time.time()
        tot_time += (t2 - t1)

    user_accuracy = accuracy_score(user_predictions, test_classes)
    task_accuracy = accuracy_score(task_predictions, task_classes)
    avr_pred_time = tot_time / len(test_data)

    print("Testing points:", len(test_data))
    print("Users correctly classified:", user_accuracy)
    print("Tasks correctly classified:", task_accuracy)
    print("Maximum number of false breaches:", max_false_negs)
    print("Maximum number of missed breaches:", max_false_pos)
    print("Avr prediction time:", avr_pred_time)

    return user_accuracy, task_accuracy, max_false_negs, max_false_pos, avr_pred_time
