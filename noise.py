from random import uniform
import numpy as np

def generate_noise(n=100):
    rand_data = np.random.uniform(low=-1, high=1, size=(n, 17))
    user_classes = [100 for _ in range(n)]
    task_classes = [100 for _ in range(n)]

    return rand_data, user_classes, task_classes

