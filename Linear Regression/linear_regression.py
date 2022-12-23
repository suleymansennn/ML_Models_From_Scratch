# Logistic Regression From Scratch
"""
Logistic regression is a type of supervised learning algorithm used for classification tasks. It is a widely used method
 for predicting the probability of a binary outcome, such as the probability of an individual having a certain disease,
 given certain characteristics (features) of that individual.

The logistic regression model estimates the probability that a given observation belongs to a certain class, based on
the values of the predictor variables. This probability is then transformed into a binary prediction using a threshold
value, usually 0.5. If the predicted probability is greater than 0.5, the model predicts the positive class (1), and if
 it is less than 0.5, the model predicts the negative class (0).

Logistic regression works by using an equation to model the relationship between the binary outcome and the input
features. The equation has the form:

y = e^(b0 + b1x1 + b2x2 + ... + bnxn) / (1 + e^(b0 + b1x1 + b2x2 + ... + bnxn))

Where y is the predicted probability of the binary outcome, x1, x2, ..., xn are the input features, and b0, b1, b2, ...,
 bn are the model coefficients. The coefficients are learned from the training data using an optimization algorithm such
  as gradient descent.

One of the key advantages of logistic regression is that it is a simple yet powerful tool for making predictions.
It can handle multiple input features, and it can also be regularized to prevent overfitting.

Another advantage of logistic regression is that it produces a probabilistic prediction, which can be useful for making
decisions in situations where the cost of making a wrong prediction is high. For example, in medical diagnosis, a
logistic regression model could be used to predict the probability of a patient having a certain disease, and a doctor
 could use this probability to decide whether to order further tests or begin treatment.

Logistic regression has some limitations, however. It assumes a linear relationship between the input features and the
 log-odds of the outcome, which may not always be the case in real-world data. It also assumes that the input features
  are independent of each other, which may not always be true.

Overall, logistic regression is a useful and widely-used tool for solving classification problems. It is simple to
implement, can handle multiple input features, and produces a probabilistic prediction, making it a valuable tool in a
variety of applications."""

import numpy as np

# Define the logistic regression model
def logistic_regression(X, y, num_iterations, learning_rate):
    # Initialize the model parameters
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    # Loop for num_iterations
    for i in range(num_iterations):
        # Calculate the predicted probabilities
        predictions = sigmoid(X.dot(weights) + bias)

        # Calculate the loss
        loss = y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        loss = np.mean(-loss)

        if i % 5000 == 0:
            print(loss)

        # Calculate the gradient of the loss with respect to the weights and bias
        gradient_weights = (predictions - y).dot(X) / m
        gradient_bias = (predictions - y).mean()

        # Update the weights and bias
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Train the logistic regression model
weights, bias = logistic_regression(X_train, y_train, num_iterations=20000, learning_rate=0.01)
# Make predictions on the dataset
predictions = sigmoid(X_test.dot(weights) + bias)
y_pred = [1 if row > 0.50 else 0 for row in predictions]