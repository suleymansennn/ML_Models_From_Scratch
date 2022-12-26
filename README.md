# Machine Learning Models 

## Logistic Regression

Logistic regression is a type of supervised learning algorithm used for classification tasks. It is a widely used method for predicting the probability of a binary outcome, such as the probability of an individual having a certain disease, given certain characteristics (features) of that individual.

The logistic regression model estimates the probability that a given observation belongs to a certain class, based on the values of the predictor variables. This probability is then transformed into a binary prediction using a threshold value, usually 0.5. If the predicted probability is greater than 0.5, the model predicts the positive class (1), and if it is less than 0.5, the model predicts the negative class (0).

Logistic regression works by using an equation to model the relationship between the binary outcome and the input features. The equation has the form:

y = e^(b0 + b1x1 + b2x2 + ... + bnxn) / (1 + e^(b0 + b1x1 + b2x2 + ... + bnxn))

Where y is the predicted probability of the binary outcome, x1, x2, ..., xn are the input features, and b0, b1, b2, ..., bn are the model coefficients. The coefficients are learned from the training data using an optimization algorithm such as gradient descent.

One of the key advantages of logistic regression is that it is a simple yet powerful tool for making predictions. It can handle multiple input features, and it can also be regularized to prevent overfitting.

Another advantage of logistic regression is that it produces a probabilistic prediction, which can be useful for making decisions in situations where the cost of making a wrong prediction is high. For example, in medical diagnosis, a logistic regression model could be used to predict the probability of a patient having a certain disease, and a doctor could use this probability to decide whether to order further tests or begin treatment.

Logistic regression has some limitations, however. It assumes a linear relationship between the input features and the log-odds of the outcome, which may not always be the case in real-world data. It also assumes that the input features are independent of each other, which may not always be true.

Overall, logistic regression is a useful and widely-used tool for solving classification problems. It is simple to implement, can handle multiple input features, and produces a probabilistic prediction, making it a valuable tool in a variety of applications.
![Screenshot 2022-12-26 183536](https://user-images.githubusercontent.com/83352965/209564164-6a5c9a5f-81dd-41de-a9f1-a8d654f5d33f.png)

## k-NN
K-NN is a machine learning algorithm that is often used for classification and regression tasks. It is called a "lazy learning" algorithm because it does not have a specific training phase, unlike other algorithms such as support vector machines or decision trees. Instead, it uses the training data to make predictions about new data points.

Here's how k-NN works:

1- The user specifies a value for k, which represents the number of nearest neighbors that will be used to make predictions.

2- The algorithm calculates the distance between the new data point and all of the points in the training dataset.

3- The algorithm selects the k training points that are closest to the new data point.

4- For classification tasks, the algorithm assigns the new data point to the class that is most common among the k nearest neighbors. For regression tasks, the algorithm calculates the average of the values of the k nearest neighbors and uses that as the predicted value for the new data point.
![Screenshot 2022-12-26 183629](https://user-images.githubusercontent.com/83352965/209564231-ddf875e5-54df-464a-9d7b-b0fae972e33f.png)
One of the key benefits of k-NN is that it is simple and easy to implement. It is also highly flexible, as the value of k can be easily adjusted to improve the performance of the algorithm.

Overall, k-NN is a useful and widely-used algorithm that can provide good results in many applications. It is especially useful for classification tasks, and it is often used in fields such as medical diagnosis and image recognition

## Decision Tree

CART is a decision tree learning algorithm that is used for classification and regression tasks. It works by recursively splitting the training data into smaller subsets based on the values of the features, with the goal of creating homogeneous subsets (i.e., subsets where the observations belong to the same class or have similar values).

To implement CART in Python, you would need to follow the following steps:

Define a class for the decision tree, which will store the information about the structure of the tree, such as the splitting feature and the threshold value used for splitting, as well as the left and right child nodes.

Define a function for splitting the data into two subsets based on the values of a given feature and a threshold value.

Define a function for calculating the metric used to evaluate the quality of the split, such as the Gini index or the entropy.

Define a function for finding the best split by iterating over all features and threshold values, and selecting the split that maximizes the metric.

Define a function for building the decision tree by recursively splitting the data until a stopping criterion is reached, such as a maximum depth or a minimum number of observations in a leaf node.

Define a function for making predictions using the decision tree, which involves traversing the tree and making a prediction based on the values of the features at each node.

![Decision-Tree-Diagram-Example-MindManager-Blog](https://user-images.githubusercontent.com/83352965/209564357-a7259ea7-71fa-4fac-bb35-9b7284ad9f8e.png)
