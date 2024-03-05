import task_model, user_model

task_model.train("rf", 1)

tasks = ["abc", "cir", "star", "www", "xyz"]
test_data, test_classes, task_classes = user_model.train(["rf"], "u1", tasks, 1)

user_model.test("u1", "rf", test_data, test_classes, task_classes, 1)
