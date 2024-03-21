import task_model
import user_model


def bulk_test():
    results = [{}, {}]
    users = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
    methods = ["svm", "rf", "knn", "dt"]
    tasks = ["abc", "cir", "star", "www", "xyz"]
    for hou in range(2):

        for m in methods:
            results[hou][m] = {}
            results[hou][m]["Task Model Train Time"] = task_model.train(m, hou)

            train_time_total = 0
            user_acc_total = 0
            task_acc_total = 0
            max_neg_overall = 0
            max_pos_overall = 0
            pred_time_total = 0
            for u in users:
                test_data, test_classes, task_classes, train_time_one_user = user_model.train([m], u, tasks, hou)
                train_time_total += train_time_one_user

                user_accuracy, task_accuracy, max_false_negs, max_false_pos, avr_pred_time = user_model.test(u, m,
                                                                                                             test_data,
                                                                                                             test_classes,
                                                                                                             task_classes,
                                                                                                             hou)
                user_acc_total += user_accuracy
                task_acc_total += task_accuracy
                pred_time_total += avr_pred_time
                if max_false_negs > max_neg_overall:
                    max_neg_overall = max_false_negs
                if max_false_pos > max_pos_overall:
                    max_pos_overall = max_false_pos

            results[hou][m]["User Models Average Train Time"] = train_time_total / len(users)
            results[hou][m]["User Prediction Accuracy"] = user_acc_total / len(users)
            results[hou][m]["Task Prediction Accuracy"] = task_acc_total / len(users)
            results[hou][m]["Average Prediction Time"] = pred_time_total / len(users)
            results[hou][m]["Max Number of Consecutive False Breaches"] = max_neg_overall
            results[hou][m]["Max Number of Consecutive Missed Breaches"] = max_pos_overall

    with open("results.txt", "w") as f:
        for hou in range(2):
            if hou == 0:
                f.write("HAPTICS\n")
            else:
                f.write("UR3E\n")

            for m, data in results[hou].items():
                f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                f.write(f"{m}\n")
                for metric, value in data.items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n")

            f.write("=======================================================\n")


bulk_test()
