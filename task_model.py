from feature_extraction import extract
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from joblib import dump
from dtaidistance import dtw


def train(method, haptics_or_ur3e=1, interval=100, verbose=True, paras=[]):
    tasks = ["abc", "cir", "star", "www", "xyz"]
    classes = {"abc": 0, "cir": 1, "star": 2, "www": 3, "xyz": 4}
    users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]

    train_data = []
    train_classes = []
    test_data = []
    test_classes = []

    for task in tasks:

        for user in users:
            train, test = extract(user, task, haptics_or_ur3e, interval=interval)
            train_data.extend(train)
            test_data.extend(test)
            train_classes.extend([classes[task]] * len(train))
            test_classes.extend([classes[task]] * len(test))
    if verbose:
        print("Training {m} model with {n} data points".format(m=method, n=len(train_data)))

    if method == "svm":
        model = SVC(kernel="sigmoid", gamma="scale")
    elif method == "rf":
        model = RandomForestClassifier(criterion="entropy", max_features="log2", n_estimators=100)
    elif method == "knn":
        model = KNeighborsClassifier(weights="distance", n_neighbors=15, algorithm="auto", p=1)
    elif method == "dt":
        model = DecisionTreeClassifier(criterion="gini", splitter="best", max_features=None)
    else:
        raise Exception("Method must be 'svm', 'rf', 'knn', or 'dt'")

    t1 = time.time()
    model.fit(train_data, train_classes)
    t2 = time.time()

    time_to_train = t2 - t1
    if verbose:
        print("Time to train:", time_to_train)

    name = "ur3e" if haptics_or_ur3e else "haptics"
    dump(model, "{m}_task_model_{h}.joblib".format(m=method, h=name))
    return time_to_train
