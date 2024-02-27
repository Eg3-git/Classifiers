from feature_extraction import extract
from fastdtw import fastdtw
from dtaidistance import dtw

dir1a = "3USER_10TASKS/index/ABC1/U1_abc.csv"
dir1b = "3USER_10TASKS/robot-endeffector/ABC1/U1_ABC-pos3.mat"
# dir1a = "3USER_10TASKS/index/O1/U1_o.csv"
# dir1b = "3USER_10TASKS/robot-endeffector/O1/U1_O-pos3.mat"
# dir1a = "3USER_10TASKS/index/CIRCLE1/U1_circle.csv"
# dir1b = "3USER_10TASKS/robot-endeffector/CIRCLE1/U1_circle-pos3.mat"

dir2a = "3USER_10TASKS/index/ABC1/U2_abc.csv"
dir2b = "3USER_10TASKS/robot-endeffector/ABC1/U2_ABC-pos3.mat"
# dir2a = "3USER_10TASKS/index/O1/U2_o.csv"
# dir2b = "3USER_10TASKS/robot-endeffector/O1/U2_O-pos3.mat"
# dir2a = "3USER_10TASKS/index/CIRCLE1/U2_circle.csv"
# dir2b = "3USER_10TASKS/robot-endeffector/CIRCLE1/U2_circle-pos3.mat"

dir3a = "3USER_10TASKS/index/ABC1/U3_abc.csv"
dir3b = "3USER_10TASKS/robot-endeffector/ABC1/U3_ABC-pos3.mat"


# dir3a = "3USER_10TASKS/index/O1/U3_o.csv"
# dir3b = "3USER_10TASKS/robot-endeffector/O1/U3_O-pos3.mat"
# dir3a = "3USER_10TASKS/index/CIRCLE1/U3_circle.csv"
# dir3b = "3USER_10TASKS/robot-endeffector/CIRCLE1/U3_circle-pos3.mat"

def calc_dtw(train, test, weights=[1 for _ in range(17)]):
    sum_n = []

    sum_k = 0
    for k in range(17):
        feature_weight = weights[k]
        train_seq = [o[k] for o in train]
        test_seq = [o[k] for o in test]

        dtw_distance = dtw.distance(test_seq, train_seq)
        sum_k += (dtw_distance * feature_weight)
        sum_n.append(sum_k)

    tot = 0
    for s in sum_n:
        tot += s
    return tot / len(sum_n)


def train_weights(w, weights, u, test, train1, train2, train3):
    m = -1
    mis = 0
    print("Feature", w, "Test User", u)

    weights[w] = 0
    r1_1 = calc_dtw(train1, test, weights)
    r2_1 = calc_dtw(train2, test, weights)
    r3_1 = calc_dtw(train3, test, weights)

    weights[w] = 1
    r1_2 = calc_dtw(train1, test, weights)
    r2_2 = calc_dtw(train2, test, weights)
    r3_2 = calc_dtw(train3, test, weights)

    dif1 = r1_2 - r1_1
    dif2 = r2_2 - r2_1
    dif3 = r3_2 - r3_1

    if u == 1:
        return min(dif2, dif3) / dif1
    elif u == 2:
        return min(dif1, dif3) / dif2
    else:
        return min(dif1, dif2) / dif3

    # print("U1, w0: {n1}, w1: {n2}, Dif: {d}".format(n1=r1_1, n2=r1_2, d=r1_2-r1_1))
    # print("U2, w0: {n1}, w1: {n2}, Dif: {d}".format(n1=r2_1, n2=r2_2, d=r2_2-r2_1))
    # print("U3, w0: {n1}, w1: {n2}, Dif: {d}".format(n1=r3_1, n2=r3_2, d=r3_2-r3_1))
    # print()


U1_train, U1_test = extract(dir1a, dir1b, flatten=True, interval=100)
U2_train, U2_test = extract(dir2a, dir2b, flatten=True, interval=100)
U3_train, U3_test = extract(dir3a, dir3b, flatten=True, interval=100)

# U1_train = U1[:int((len(U1)*0.75))]
# U1_test = U1[int((len(U1)*0.75)):]

# U2_train = U2[:int((len(U2)*0.75))]
# U2_test = U2[int((len(U2)*0.75)):]

# U3_train = U3[:int((len(U3)*0.75))]
# U3_test = U3[int((len(U3)*0.75)):]

ws = [0.09, 0.08, 0.06, 0.06, 0.04, 0.06, 0.05, 0.05, 0.07, 0.06, 0.07, 0.05, 0.06, 0.05, 0.06, 0.05, 0.04]
accuracy = 0

U1_scores = []
print(len(U1_test))
for x in range(100, len(U1_test), 100):
    print(x)
    U1_scores.append(calc_dtw(U1_train, U1_test[x - 100:x], ws))

U2_scores = []
for x in range(100, len(U2_test), 100):
    U2_scores.append(calc_dtw(U1_train, U2_test[x - 100:x], ws))

U3_scores = []
for x in range(100, len(U3_test), 100):
    U3_scores.append(calc_dtw(U1_train, U3_test[x - 100:x], ws))

print("U1 scores:", U1_scores)
print("U2 scores:", U2_scores)
print("U3 scores:", U3_scores)
