import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from file_selection import file_selection
from joblib import dump
from feature_extraction import extract

from pyod.models.knn import KNN
from pyod.utils import evaluate_print

files = file_selection(["*"], ["o"])
dir1A, dir1B = files[("U1", "o")]
dir2A, dir2B = files[("U2", "o")]
dir3A, dir3B = files[("U3", "o")]

intervals = [100]

for t in intervals:
    X_train, X_test = extract(dir1A, dir1B, t)
    Y_train, Y_test = extract(dir2A, dir2B, t)
    Z_train, Z_test = extract(dir3A, dir3B, t)
    train_data = X_train + Y_train + Z_train
    train_classes = [0 for _ in X_train] + [1 for _ in Y_train] + [1 for _ in Z_train]
    test_data = X_test + Y_test + Z_test
    test_classes = [0 for _ in X_test] + [1 for _ in Y_test] + [1 for _ in Z_test]

    model = KNN()
    model.fit(train_data)

    y_pred = model.labels_
    y_scores = model.decision_scores_

    preds = model.predict(test_data)
    scores = model.decision_function(test_data)

    print("\nOn Training Data:")
    evaluate_print('KNN', train_classes, y_scores)
    print("\nOn Test Data:")
    evaluate_print('KNN', test_classes, scores)
print("saving")
dump(model, "ABC_model_knn.joblib")
print("done")
