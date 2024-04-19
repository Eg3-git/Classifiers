import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display_cm(cm, xlabels, ylabels):
    sns.set(font_scale=3)
    cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cmn, annot=True, fmt='.1f', xticklabels=xlabels, yticklabels=ylabels, cbar=False)
    for t in ax.texts: t.set_text(t.get_text() + "%")
    plt.ylabel('Actual Task')
    plt.xlabel('Predicted Task')
    plt.show(block=False)


cm_rf_nott = np.array([[22715, 792, 1606, 792, 715],
               [704, 20317, 550, 308, 286],
               [1958, 693, 17886, 495, 605],
               [660., 495., 264., 15059., 825.],
               [243., 825., 781., 539., 13365.]])

x = ["ABC", "CIRCLE", "STAR", "WWW", "XYZ"]

cm_rf_tt = np.array([[2.6598e+04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.2000e+01],
                [9.9000e+01, 2.1923e+04, 1.1000e+02, 0.0000e+00, 3.3000e+01],
                [0.0000e+00, 4.4000e+01, 2.1461e+04, 2.2000e+01, 1.1000e+02],
                [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7303e+04, 0.0000e+00],
                [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6753e+04]])

cm_knn_nott = np.array([[16819.,  2915.,  3267.,  1859.,  1760.],
 [ 1408., 18590.,   814.,   737.,   616.],
 [ 4620.,  1595., 12760.,  1342.,  1320.],
 [ 3619.,  1001.,  1529., 10010.,  1144.],
 [ 3487.,  1232.,  1903.,  1452.,  8679.]])
cm_knn_tt = np.array([[2.6598e+04, 0.0000e+00, 0.0000e+00, 2.2000e+01, 0.0000e+00],
 [0.0000e+00, 2.2165e+04, 0.0000e+00, 0.0000e+00, 0.0000e+00],
 [1.6170e+03, 2.2000e+01, 1.9998e+04, 0.0000e+00, 0.0000e+00],
 [1.2100e+02, 7.7000e+01, 0.0000e+00, 1.7083e+04, 2.2000e+01],
 [3.9600e+02, 0.0000e+00, 0.0000e+00, 7.2600e+02, 1.5631e+04]])

cm_dt_nott = np.array([[18843.,  1188.,  3212.,  1661.,  1716.],
 [ 1199., 18436.,   825.,   825.,   880.],
 [ 2981.,  1133., 15301.,  1144.,  1078.],
 [ 1331.,   671.,   968., 12881.,  1452.],
 [ 1870.,   924., 1210.,  1397., 11352.]])
cm_dt_tt = np.array([[2.6620e+04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
 [6.6000e+01, 2.1934e+04, 6.6000e+01, 0.0000e+00, 9.9000e+01],
 [3.3000e+01, 6.6000e+01, 2.1439e+04, 2.2000e+01, 7.7000e+01],
 [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.7303e+04, 0.0000e+00],
 [1.4300e+02, 4.4000e+01, 4.4000e+01, 0.0000e+00, 1.6522e+04]])


display_cm(cm_rf_nott, x, x)
display_cm(cm_rf_tt, x, x)
display_cm(cm_knn_nott, x, x)
display_cm(cm_knn_tt, x, x)
display_cm(cm_dt_nott, x, x)
display_cm(cm_dt_tt, x, x)



def plot_bar(x, ys, labels, x_label, y_label, title):
    w = np.arange(len(x))
    bar_width = 0.25
    colours = ["b", "g", "r", "y"]

    for i in range(len(ys)):
        plt.bar(w + (i / (2 * len(ys))) - (bar_width / len(ys)), ys[i], width=bar_width, color=colours[i],
                align='center', label=labels[i])

    # Adding labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(w, x)
    plt.legend()

    # Display the plot
    plt.show()


x = ["SVM", "RF", "KNN", "DT"]
ys = [[0.05, 1, 0.4, 0.9], [0.05, 0.9, 0.2, 0.7]]

ls = ["With RF task model", "With DT task model"]
xl = "User Model"
yl = "Accepted Random Data Points (%)"

# plot_bar(x, ys, ls, xl, yl, "Verification Rates for optimal time interval")

# display_cm(cm, x, x)
