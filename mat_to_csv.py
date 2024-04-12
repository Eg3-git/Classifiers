import csv
from scipy.io import loadmat
import numpy as np

tasks = ["abc", "circle", "star", "w", "z"]
new_tasks = ["abc", "cir", "star", "www", "xyz"]
users = ["U1", "U2", "U3"]
new_users = ["u9", "u10", "u11"]

index_dir = "ROBOT_USER_DATA/index/"
data_dir = "ROBOT_USER_DATA/robot-endeffector/"
target_dir = "ROBOT_USER_DATA/converted/"

for t in range(len(tasks)):
    for u in range(len(users)):
        with open(f"{index_dir}{tasks[t]}/{users[u]}_{tasks[t]}.csv", newline='') as f:
            all_indices = csv.reader(f)
            #indices = [next(all_indices) for _ in range(len(all_indices))]
            indices = [row for row in all_indices]
            mat = loadmat(f"{data_dir}{tasks[t]}/{users[u]}_{tasks[t]}-pos3.mat")['pos3']
            mat_adjusted = np.insert(mat, 0, [0 for _ in mat], axis=1)

            for i in range(len(indices)):
                a = int(indices[i][0])
                b = int(indices[i][1])
                with open(f"{target_dir}{new_users[u]}{new_tasks[t]}{i+1}_ur3e_end_effectors_pose.csv", "w") as o:
                    writer = csv.writer(o)
                    writer.writerow(["time","x","y","z"])

                    while a < b:
                        writer.writerow(mat_adjusted[a])
                        a += 1

