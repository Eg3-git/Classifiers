import tests


def file_selection(users, tests):
    out = {}
    for user in users:
        for test in tests:
            if test == "abc":
                if user == "*":
                    out[("U1", test)] = ("3USER_10TASKS/index/ABC1/U1_abc.csv", "3USER_10TASKS/robot-endeffector/ABC1/U1_ABC-pos3.mat")
                    out[("U2", test)] = (
                    "3USER_10TASKS/index/ABC1/U2_abc.csv", "3USER_10TASKS/robot-endeffector/ABC1/U2_ABC-pos3.mat")
                    out[("U3", test)] = (
                    "3USER_10TASKS/index/ABC1/U3_abc.csv", "3USER_10TASKS/robot-endeffector/ABC1/U3_ABC-pos3.mat")

                else:
                    out[(user, test)] = ("3USER_10TASKS/index/ABC1/" + user + "_abc.csv", "3USER_10TASKS/robot-endeffector/ABC1/"+user+"_ABC-pos3.mat")

            if test == "o":
                if user == "*":
                    out[("U1", test)] = ("3USER_10TASKS/index/O1/U1_o.csv", "3USER_10TASKS/robot-endeffector/O1/U1_O-pos3.mat")
                    out[("U2", test)] = (
                        "3USER_10TASKS/index/O1/U2_o.csv", "3USER_10TASKS/robot-endeffector/O1/U2_O-pos3.mat")
                    out[("U3", test)] = (
                    "3USER_10TASKS/index/O1/U3_o.csv", "3USER_10TASKS/robot-endeffector/O1/U3_O-pos3.mat")

                else:
                    out[(user, test)] = ("3USER_10TASKS/index/O1/" + user + "_o.csv", "3USER_10TASKS/robot-endeffector/O1/"+user+"_O-pos3.mat")

            if test == "circle":
                if user == "*":
                    out[("U1", test)] = ("3USER_10TASKS/index/CIRCLE1/U1_circle.csv", "3USER_10TASKS/robot-endeffector/CIRCLE1/U1_circle-pos3.mat")
                    out[("U2", test)] = (
                    "3USER_10TASKS/index/CIRCLE1/U2_circle.csv", "3USER_10TASKS/robot-endeffector/CIRCLE1/U2_circle-pos3.mat")
                    out[("U3", test)] = (
                    "3USER_10TASKS/index/CIRCLE1/U3_circle.csv", "3USER_10TASKS/robot-endeffector/CIRCLE1/U3_circle-pos3.mat")

                else:
                    out[(user, test)] = ("3USER_10TASKS/index/CIRCLE1/" + user + "_circle.csv", "3USER_10TASKS/robot-endeffector/CIRCLE1/"+user+"_circle-pos3.mat")

    return out
