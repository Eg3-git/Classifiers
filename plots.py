import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_cm(cm, xlabels, ylabels):
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=xlabels, yticklabels=ylabels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)


cm = np.array([[6224, 1464, 904, 600, 1552],
               [784, 9016, 1088, 440, 808],
               [1264, 1072, 6664, 504, 776],
               [560, 792, 440, 6944, 736],
               [1216, 816, 968, 744, 5728]])

x = ["ABC", "CIRCLE", "STAR", "WWW", "XYZ"]


def plot_bar(x, ys, labels, x_label, y_label, title):
    w = np.arange(len(x))
    bar_width = 0.25
    colours = ["b", "g", "r", "y"]

    for i in range(len(ys)):
        plt.bar(w - bar_width, ys[i], width=bar_width, color=colours[i], align='center', label=labels[i])

    # Adding labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(w, x)
    plt.legend()

    # Display the plot
    plt.show()


x = ["SVM", "RF", "KNN", "DT"]
ys = [[3.33, 3.33, 1.67, 10], [1.25, 3.33, 3.33, 10], [1.25, 3.33, 3.33, 10]]

ls = ["Same Classifier for task and user model", "With RF task model", "With DT task model"]
xl = "Classifier"
yl = "Verifications per second"

plot_bar(x, ys, ls, xl, yl, "Verification Rates for optimal time interval")

# display_cm(cm, x, x)
