import numpy as np


def compute_cost_with_regularization_test(target):
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
    lambd = 0.1

    m = Y.shape[1]
    cross_entropy_cost = (1. / m) * np.sum(
        np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    )
    l2_cost = (1. / m) * (lambd / 2) * (
        np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
    )
    expected_cost = cross_entropy_cost + l2_cost

    cost = target(a3, Y, parameters, lambd)

    assert np.isclose(cost, expected_cost, atol=1e-6), \
        f"Wrong cost. Expected: {expected_cost}, got: {cost}"

    print("\033[92mAll tests passed.")


def backward_propagation_with_regularization_test(target):
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

    lambd = 0.7
    m = X.shape[1]

    dZ3 = A3 - Y
    dW3_expected = 1. / m * np.dot(dZ3, A2.T) + (lambd / m) * W3
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2_expected = 1. / m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1_expected = 1. / m * np.dot(dZ1, X.T) + (lambd / m) * W1

    grads = target(X, Y, cache, lambd)

    assert np.allclose(grads["dW1"], dW1_expected, atol=1e-6), "Wrong values for dW1"
    assert np.allclose(grads["dW2"], dW2_expected, atol=1e-6), "Wrong values for dW2"
    assert np.allclose(grads["dW3"], dW3_expected, atol=1e-6), "Wrong values for dW3"

    print("\033[92mAll tests passed.")


def forward_propagation_with_dropout_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    W1 = np.random.randn(20, 2)
    b1 = np.random.randn(20, 1)
    W2 = np.random.randn(3, 20)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    keep_prob = 0.7

    A3, cache = target(X, parameters, keep_prob)

    assert A3.shape == (1, X.shape[1]), f"Wrong shape for A3: {A3.shape}"
    assert np.all((A3 >= 0) & (A3 <= 1)), "A3 should contain values between 0 and 1 (sigmoid output)"

    D1 = cache[1]
    D2 = cache[6]
    assert set(np.unique(D1)).issubset({0, 1}), "D1 should only contain 0s and 1s"
    assert set(np.unique(D2)).issubset({0, 1}), "D2 should only contain 0s and 1s"

    print("\033[92mAll tests passed.")


def backward_propagation_with_dropout_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = np.array([[1, 0, 1]])
    W1 = np.random.randn(2, 2)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)

    keep_prob = 0.8

    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    np.random.seed(1)
    D1 = (np.random.rand(*A1.shape) < keep_prob).astype(int)
    A1 = A1 * D1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    D2 = (np.random.rand(*A2.shape) < keep_prob).astype(int)
    A2 = A2 * D2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = 1 / (1 + np.exp(-Z3))
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    m = X.shape[1]
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    dA2 = np.dot(W3.T, dZ3)
    dA2_expected = (dA2 * D2) / keep_prob
    dZ2 = np.multiply(dA2_expected, np.int64(A2 > 0))
    dA1 = np.dot(W2.T, dZ2)
    dA1_expected = (dA1 * D1) / keep_prob

    grads = target(X, Y, cache, keep_prob)

    assert np.allclose(grads["dA1"], dA1_expected, atol=1e-6), "Wrong values for dA1"
    assert np.allclose(grads["dA2"], dA2_expected, atol=1e-6), "Wrong values for dA2"

    print("\033[92mAll tests passed.")
