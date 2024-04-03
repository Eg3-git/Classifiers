import csv

import numpy as np
from matplotlib import pyplot as plt


def plot_trajectory():

    tasks = ["abc", "cir", "star", "www", "xyz"]
    base_dir = "Data Collection/u1/u1"
    dir_end = "_ur3e_end_effectors_pose.csv"

    for task in tasks:
        f_dir = "{a}{t}1{b}".format(a=base_dir, t=task, b=dir_end)
        fig = plt.figure()

        ax = plt.axes(projection='3d')



        with open(f_dir, newline='') as f:
            data = list(csv.reader(f))[1:]
            np_data = np.array(data, dtype=float)[:, 1:]

            prev_point = data[0]

            ax.plot3D(np_data[:, 0], np_data[:, 1], np_data[:, 2])
            ax.set_title(task)
            plt.show()


plot_trajectory()