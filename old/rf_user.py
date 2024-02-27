from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from feature_extraction import extract
from joblib import dump, load
import time

intervals = [100]


def train(users, tasks):
    test_data = []
    test_classes = []
    task_classes = []
    t = 0
    tot_time = 0
    for task in tasks:
        root_index = "ROBOT_USER_DATA/index/{t}/".format(t=task)
        root_endef = "ROBOT_USER_DATA/robot-endeffector/{t}/".format(t=task)

        dir1a = "{r}U1_{t}.csv".format(r=root_index, t=task)
        dir1b = "{r}U1_{t}-pos3.mat".format(r=root_endef, t=task)

        dir2a = "{r}U2_{t}.csv".format(r=root_index, t=task)
        dir2b = "{r}U2_{t}-pos3.mat".format(r=root_endef, t=task)

        dir3a = "{r}U3_{t}.csv".format(r=root_index, t=task)
        dir3b = "{r}U3_{t}-pos3.mat".format(r=root_endef, t=task)

        X_train, X_test = extract(dir1a, dir1b)
        Y_train, Y_test = extract(dir2a, dir2b)
        Z_train, Z_test = extract(dir3a, dir3b)
        train_data = X_train + Y_train + Z_train
        train_classes = [0 for _ in X_train] + [1 for _ in Y_train] + [1 for _ in Z_train]
        test_data += (X_test + Y_test + Z_test)
        test_classes += ([0 for _ in X_test] + [1 for _ in Y_test] + [1 for _ in Z_test])
        task_classes += ([t for _ in X_test + Y_test + Z_test])
        t += 1

        print("training", task)
        model = RandomForestClassifier()
        t1 = time.time()
        model.fit(train_data, train_classes)
        t2 = time.time()

        tot_time += (t2 - t1)
        dump(model, "models/rf/{t}/rf_U1_{t}.joblib".format(t=task))
    print("Avr train time:", tot_time / len(train_data))
    return test_data, test_classes, task_classes


def test(user, test_data, test_classes, task_classes):
    task_model = load("../task_model_rf.joblib")
    user_predictions = []
    task_predictions = []
    tot_time = 0

    for datum in test_data:
        t1 = time.time()
        predicted_task = task_model.predict([datum])[0]
        task_predictions.append(predicted_task)
        user_model = load("models/rf/{t}/rf_{u}_{t}.joblib".format(u=user, t=tasks[predicted_task]))
        user_predictions.append(user_model.predict([datum]))
        t2 = time.time()
        tot_time += (t2 - t1)

    print("Testing points:", len(test_data))
    print("Users correctly classified:", accuracy_score(user_predictions, test_classes))
    print("Tasks correctly classified:", accuracy_score(task_predictions, task_classes))
    print("Avr prediction time:", tot_time / len(test_data))


tasks = ["abc", "circle", "o", "p2p2", "push", "s", "star", "tri", "w", "z"]

data, classes, task_classes = train([], tasks)
test("U1", data, classes, task_classes)
