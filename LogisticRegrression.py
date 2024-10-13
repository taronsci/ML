import numpy as np


def logistic_cost(X, Y, theta):
    """
    Computes the cost function for logistic regression.
    """
    # Number of samples
    m = len(Y)

    # Hypothesis function (sigmoid function)
    h_theta = 1 / (1 + np.exp(-X.dot(theta)))

    # Compute the cost function
    cost = -(1 / m) * np.sum(Y * np.log(h_theta) + (1 - Y) * np.log(1 - h_theta))

    return cost


def gradient_descent(X, Y, theta, alpha, epochs):
    """
    Performs gradient descent to optimize theta.
    """
    # Number of samples
    m = len(Y)

    # Initialize a list to store the history of cost values
    cost_history = []

    for i in range(epochs):
        # Hypothesis function (sigmoid function)
        h_theta = 1 / (1 + np.exp(-X.dot(theta)))

        # Compute the gradient
        gradient = (1 / m) * X.T.dot(h_theta - Y)

        # Update theta
        theta -= alpha * gradient

        # Compute the cost
        cost = logistic_cost(X, Y, theta)
        # Store the cost after each iteration
        cost_history.append(cost)

    return theta, cost_history
