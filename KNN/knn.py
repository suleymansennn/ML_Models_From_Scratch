# k-NN From Scratch
"""
K-NN is a machine learning algorithm that is often used for classification and regression tasks. It is called a
"lazy learning" algorithm because it does not have a specific training phase, unlike other algorithms such as support
vector machines or decision trees. Instead, it uses the training data to make predictions about new data points.

Here's how k-NN works:

1- The user specifies a value for k, which represents the number of nearest neighbors that will be used to make
predictions.

2- The algorithm calculates the distance between the new data point and all of the points in the training dataset.

3- The algorithm selects the k training points that are closest to the new data point.

4- For classification tasks, the algorithm assigns the new data point to the class that is most common among the k
nearest neighbors. For regression tasks, the algorithm calculates the average of the values of the k nearest neighbors
and uses that as the predicted value for the new data point.

One of the key benefits of k-NN is that it is simple and easy to implement. It is also highly flexible, as the value
of k can be easily adjusted to improve the performance of the algorithm.

Overall, k-NN is a useful and widely-used algorithm that can provide good results in many applications. It is
especially useful for classification tasks, and it is often used in fields such as medical diagnosis and image
recognition.
"""
import numpy as np
# define a function to calculate the Euclidean distance between two data points
def euclidean_distance(x1, x2):
     return np.sqrt(np.sum((x1 - x2)**2))

def most_common_label(labels):
    return pd.Series(labels).mode()[0]

def KNN_Clasifier(X_train, y_train, X_test, k_neighbours=3):
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    # initialize a list to store the predicted labels
    y_pred = []
    # loop through each data point in the test set
    for x in X_test:
        distances = []
        for i in range(X_train.shape[0]):
                # calculate the distance between the test point and each training point
                distance = euclidean_distance(x, X_train[i])
                distances.append((distance, y_train[i]))
        # sort the distances in ascending order and get the indices of the k-nearest neighbors
        k_nearest = sorted(distances)[:k_neighbours]
        # get the labels of the k-nearest neighbors
        labels = [label for _, label in k_nearest]
        # use the mode to predict the label for the test point
        y_pred.append(most_common_label(labels))
    return y_pred


y_pred = KNN_Clasifier(X_train, y_train, X_test, 3)