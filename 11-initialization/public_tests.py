import numpy as np


def initialize_parameters_zeros_test(target):
    layer_dims = [3, 2, 1]
    parameters = target(layer_dims)

    assert isinstance(parameters, dict), "Output must be a dict"
    assert len(parameters) == 2 * (len(layer_dims) - 1), \
        f"Number of parameters must be {2 * (len(layer_dims) - 1)}, got {len(parameters)}"

    for l in range(1, len(layer_dims)):
        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]
        expected_W_shape = (layer_dims[l], layer_dims[l - 1])
        expected_b_shape = (layer_dims[l], 1)
        assert Wl.shape == expected_W_shape, \
            f"W{l} shape mismatch: expected {expected_W_shape}, got {Wl.shape}"
        assert bl.shape == expected_b_shape, \
            f"b{l} shape mismatch: expected {expected_b_shape}, got {bl.shape}"
        assert np.all(Wl == 0), f"W{l} should be all zeros"
        assert np.all(bl == 0), f"b{l} should be all zeros"

    print("\033[92mAll tests passed.")


def initialize_parameters_random_test(target):
    layer_dims = [3, 2, 1]
    parameters = target(layer_dims)

    assert isinstance(parameters, dict), "Output must be a dict"
    assert len(parameters) == 2 * (len(layer_dims) - 1), \
        f"Number of parameters must be {2 * (len(layer_dims) - 1)}, got {len(parameters)}"

    for l in range(1, len(layer_dims)):
        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]
        expected_W_shape = (layer_dims[l], layer_dims[l - 1])
        expected_b_shape = (layer_dims[l], 1)
        assert Wl.shape == expected_W_shape, \
            f"W{l} shape mismatch: expected {expected_W_shape}, got {Wl.shape}"
        assert bl.shape == expected_b_shape, \
            f"b{l} shape mismatch: expected {expected_b_shape}, got {bl.shape}"
        assert np.all(bl == 0), f"b{l} should be initialized to zeros"
        assert not np.all(Wl == 0), f"W{l} should not be all zeros (random init)"

    print("\033[92mAll tests passed.")


def initialize_parameters_he_test(target):
    layer_dims = [2, 4, 1]
    parameters = target(layer_dims)

    assert isinstance(parameters, dict), "Output must be a dict"
    assert len(parameters) == 2 * (len(layer_dims) - 1), \
        f"Number of parameters must be {2 * (len(layer_dims) - 1)}, got {len(parameters)}"

    expected = {
        "W1": np.array([[1.78862847, 0.43650985],
                         [0.09649747, -1.8634927],
                         [-0.2773882, -0.35475898],
                         [-0.08274148, -0.62700068]]),
        "b1": np.zeros((4, 1)),
        "W2": np.array([[-0.03098412, -0.33744411, -0.92904268, 0.62552248]]),
        "b2": np.zeros((1, 1)),
    }

    for key in expected:
        assert key in parameters, f"Missing key: {key}"
        assert np.allclose(parameters[key], expected[key], atol=1e-6), \
            f"Wrong values for {key}. Expected:\n{expected[key]}\nGot:\n{parameters[key]}"

    print("\033[92mAll tests passed.")