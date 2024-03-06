import task_model, user_model, noise

#task_model.train("knn", 1)

tasks = ["abc", "cir", "star", "www", "xyz"]
test_data, test_classes, task_classes = user_model.train(["knn"], "u1", tasks, 1)

#test_data, test_classes, task_classes = noise.generate_noise()

user_model.test("u1", "knn", test_data, test_classes, task_classes, 1)
