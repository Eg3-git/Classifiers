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

display_cm(cm, x, x)
