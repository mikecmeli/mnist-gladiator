from sklearn.tree import DecisionTreeClassifier


def decision_tree(train_x, train_y, test_x, test_y, **kwargs):
    classifier = DecisionTreeClassifier()
    classifier.fit(train_x, train_y)

    pred_y = classifier.predict(test_x)

    equal_indices = test_y == pred_y
    equal_amount = equal_indices.sum()
    return (equal_amount / len(test_y), pred_y)
