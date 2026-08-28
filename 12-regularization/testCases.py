import numpy as np


def compute_cost_with_regularization_test_case():
    np.random.seed(1)
    Y = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    a3 = np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])
    return a3, Y, parameters


def backward_propagation_with_regularization_test_case():
    np.random.seed(1)
    X = np.random.randn(3, 5)
    Y = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = 1 / (1 + np.exp(-Z3))

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return X, Y, cache


def forward_propagation_with_dropout_test_case():
    np.random.seed(1)
    X = np.random.randn(2, 3)
    W1 = np.random.randn(20, 2)
    b1 = np.random.randn(20, 1)
    W2 = np.random.randn(3, 20)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return X, parameters


def backward_propagation_with_dropout_test_case():
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = np.array([[1, 0, 1]])
    W1 = np.random.randn(2, 2)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    np.random.seed(1)
    D1 = (np.random.rand(*A1.shape) < 0.8).astype(int)
    A1 = A1 * D1 / 0.8

    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    D2 = (np.random.rand(*A2.shape) < 0.8).astype(int)
    A2 = A2 * D2 / 0.8

    Z3 = np.dot(W3, A2) + b3
    A3 = 1 / (1 + np.exp(-Z3))

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return X, Y, cache
