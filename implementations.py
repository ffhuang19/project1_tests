import numpy as np


def compute_mse_loss(y, tx, w):
    """Compute the mean squared error loss with a factor of 0.5."""
    e = y - tx.dot(w)
    return (e.dot(e)) / (2 * len(y))


def compute_logistic_loss(y, tx, w):
    """Compute the logistic loss using logaddexp for numerical stability."""
    s = tx.dot(w)
    return np.mean(np.logaddexp(0, s) - y * s)


def sigmoid(t):
    """Apply the sigmoid function element-wise to t."""
    return 1.0 / (1 + np.exp(-t))


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w.copy()
    n = len(y)
    for _ in range(max_iters):
        gradient = tx.T.dot(tx.dot(w) - y) / n
        w -= gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    w = initial_w.copy()
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)
        xi, yi = tx[i], y[i]
        gradient = (xi.dot(w) - yi) * xi
        w -= gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations with L2 regularization."""
    n, m = tx.shape[0], tx.shape[1]
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.eye(m)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    w = initial_w.copy()
    n = len(y)
    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        gradient = tx.T.dot(pred - y) / n
        w -= gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent with L2 penalty."""
    w = initial_w.copy()
    n = len(y)
    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        gradient = tx.T.dot(pred - y) / n + 2 * lambda_ * w
        w -= gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
