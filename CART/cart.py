# CART
"""CART is a decision tree learning algorithm that is used for classification and regression tasks. It works by
recursively splitting the training data into smaller subsets based on the values of the features, with the goal of
 creating homogeneous subsets (i.e., subsets where the observations belong to the same class or have similar values).

To implement CART in Python, you would need to follow the following steps:

Define a class for the decision tree, which will store the information about the structure of the tree, such as the
splitting feature and the threshold value used for splitting, as well as the left and right child nodes.

Define a function for splitting the data into two subsets based on the values of a given feature and a threshold value.

Define a function for calculating the metric used to evaluate the quality of the split, such as the Gini index or the
entropy.

Define a function for finding the best split by iterating over all features and threshold values, and selecting the
split that maximizes the metric.

Define a function for building the decision tree by recursively splitting the data until a stopping criterion is
reached, such as a maximum depth or a minimum number of observations in a leaf node.

Define a function for making predictions using the decision tree, which involves traversing the tree and making a
prediction based on the values of the features at each node."""

# Import necessary libraries
from collections import Counter
from math import log

# Create a class for the Decision Tree Classifier
class DecisionTreeClassifier:

    # Define a method for training the model
    def fit(self, X, y):

        # Get the unique values and the counts of each class in the target variable
        self.classes = Counter(y)

        # Calculate the entropy of the target variable
        self.entropy = self._calculate_entropy(y)

        # Get the number of features in the data
        self.num_features = len(X[0])

        # Create a dictionary to store the best split for each feature
        self.best_splits = {}

        # Iterate over each feature
        for feature_index in range(self.num_features):

            # Calculate the best split for the current feature
            best_split = self._calculate_best_split(X, y, feature_index)

            # Update the best splits dictionary with the best split for the current feature
            self.best_splits[feature_index] = best_split

    # Define a method for making predictions
    def predict(self, X):

        # Create a list to store the predicted classes
        y_pred = []

        # Iterate over each sample in the data
        for sample in X:

            # Create a variable to store the current class
            current_class = None

            # Iterate over each feature and its best split
            for feature_index, (split_value, branches) in self.best_splits.items():

                # Check which branch of the split the current sample belongs to
                if sample[feature_index] <= split_value:
                    current_class = branches[0]
                else:
                    current_class = branches[1]

            # Add the predicted class for the current sample to the list of predicted classes
            y_pred.append(current_class)

        # Return the list of predicted classes
        return y_pred

    # Define a method for calculating the entropy of the target variable
    def _calculate_entropy(self, y):

        # Initialize the entropy to 0
        entropy = 0

        # Iterate over each unique class in the target variable
        for class_value, count in self.classes.items():

            # Calculate the probability of the current class
            prob = count / len(y)

            # Update the entropy
            entropy -= prob * log(prob, 2)

        # Return the entropy
        return entropy

    # Define a method for calculating the best split for a given feature
    def _calculate_best_split(self, X, y, feature_index):

        # Get the values for the current feature
        feature_values = [sample[feature_index] for sample in X]

        # Sort the values for the current feature
        sorted_values = sorted(feature_values)

        # Create a list to store the potential splits
        potential_splits = []

        # Iterate over the values for the current feature
        for index in range(len(sorted_values) - 1):

            # Calculate the average of the current and next value
                    potential_split = (sorted_values[index] + sorted_values[index + 1]) / 2

            # Add the potential split to the list of potential splits
                    potential_splits.append(potential_split)

        # Initialize the best split and its entropy to None
        best_split = None
        lowest_entropy = None

        # Iterate over the potential splits
        for potential_split in potential_splits:

            # Calculate the entropy for the current split
            entropy = self._calculate_entropy_for_split(X, y, feature_index, potential_split)

            # Check if the current split has lower entropy than the current best split
            if lowest_entropy is None or entropy <= lowest_entropy:
                best_split = potential_split
                lowest_entropy = entropy

        # Return the best split and its entropy
        return best_split, lowest_entropy

    # Define a method for calculating the entropy of a split
    def _calculate_entropy_for_split(self, X, y, feature_index, split_value):

        # Create two lists to store the samples in each branch of the split
        left_branch = []
        right_branch = []

        # Iterate over each sample in the data
        for sample, target in zip(X, y):

            # Check which branch of the split the current sample belongs to
            if sample[feature_index] <= split_value:
                left_branch.append(target)
            else:
                right_branch.append(target)

        # Calculate the entropy of each branch of the split
        left_entropy = self._calculate_entropy(left_branch)
        right_entropy = self._calculate_entropy(right_branch)

        # Calculate the total entropy of the split
        total_entropy = (len(left_branch) / len(y)) * left_entropy + (len(right_branch) / len(y)) * right_entropy

        # Return the total entropy of the split
        return total_entropy

