from feature_extraction import extract
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from joblib import dump


def train(method, haptics_or_ur3e=0):
    tasks = ["abc", "cir", "star", "www", "xyz"]
    classes = {"abc": 0, "cir": 1, "star": 2, "www": 3, "xyz": 4}
    users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]

    train_data = []
    train_classes = []
    test_data = []
    test_classes = []

    for task in tasks:
        print(task)
        for user in users:
            train, test = extract(user, task, haptics_or_ur3e)
            train_data.extend(train)
            test_data.extend(test)
            train_classes.extend([classes[task]] * len(train))
            test_classes.extend([classes[task]] * len(test))

    print("Training {m} model with {n} data points".format(m=method, n=len(train_data)))

    if method == "svm":
        model = SVC()
    elif method == "rf":
        model = RandomForestClassifier()
    elif method == "knn":
        model = KNeighborsClassifier()
    else:
        raise Exception("Method must be 'svm', 'rf', or 'knn'")

    t1 = time.time()
    model.fit(train_data, train_classes)
    t2 = time.time()

    print("Time to train:", (t2 - t1))

    name = "ur3e" if haptics_or_ur3e else "haptics"
    dump(model, "{m}_task_model_{h}.joblib".format(m=method, h=name))
