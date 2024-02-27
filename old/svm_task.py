from feature_extraction import extract
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
from joblib import dump

tasks = ["abc", "circle", "o", "p2p2", "push", "s", "star", "tri", "w", "z"]
classes = {"abc": 0, "circle": 1, "o": 2, "p2p2": 3, "push": 4, "s": 5, "star": 6, "tri": 7, "w": 8, "z": 9}
users = ["U1", "U2", "U3"]

train_data = []
train_classes = []
test_data = []
test_classes = []

for task in tasks:
    print(task)
    for user in users:
        d1 = "ROBOT_USER_DATA/index/{t}/{u}_{t}.csv".format(t=task, u=user)
        d2 = "ROBOT_USER_DATA/robot-endeffector/{t}/{u}_{t}-pos3.mat".format(t=task, u=user)
        train, test = extract(d1, d2)
        train_data.extend(train)
        test_data.extend(test)
        train_classes.extend([classes[task]] * len(train))
        test_classes.extend([classes[task]] * len(test))

print("started", len(train_data))
model = svm.SVC(kernel="linear")
print("training")
t1 = time.time()
model.fit(train_data, train_classes)
t2 = time.time()

print("Training points", len(train_data))
print("Time to train:", (t2 - t1))
print("saving")
dump(model, "../task_model_svm.joblib")
print("done")
