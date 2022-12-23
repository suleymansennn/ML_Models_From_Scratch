# Random Forest
"""
Random Forest is an ensemble machine learning algorithm that is used for classification and regression tasks.
It is a type of decision tree algorithm that creates a set of decision trees from randomly selected subsets of the
training data, and then combines the predictions of those trees to make a final prediction.

The main advantage of Random Forest is that it can handle large datasets with high dimensionality, and it can also
handle missing data. It also has a high accuracy and is resistant to overfitting, which makes it a popular choice
for many applications.

In order to create a Random Forest model, the training data is first split into a number of subsets, and a decision
tree is trained on each subset. The decision trees are trained using a random subset of the features, and each tree
makes a prediction based on the features it was trained on. The final prediction is made by averaging the predictions
of all the decision trees.

One of the key features of Random Forest is that it allows for feature importance, which is a measure of how much a
feature contributes to the prediction made by the model. This can be useful for understanding which features are most
important in making a prediction, and for identifying potential sources of error in the model.

There are some limitations to Random Forest, however. One is that it can be computationally expensive to train and run,
especially for large datasets. Additionally, the model can be difficult to interpret, since it is made up of many
decision trees and it is not always clear how the final prediction was made.

Overall, Random Forest is a powerful and widely-used machine learning algorithm that can be applied to a variety of
tasks. Its ability to handle large datasets and missing data,as well as its high accuracy and resistance to overfitting,
make it a valuable tool for data scientists and machine learning practitioners.
"""
import numpy as np
from collections import Counter
from CART.cart import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier()
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions