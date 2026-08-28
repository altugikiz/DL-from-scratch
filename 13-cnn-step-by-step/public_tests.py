import numpy as np


def zero_pad_test(target):
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = target(x, 3)

    assert type(x_pad) == np.ndarray, "Output must be a np.ndarray"
    assert x_pad.shape == (4, 9, 9, 2), f"Wrong shape: {x_pad.shape} != (4, 9, 9, 2)"
    assert np.allclose(x_pad[0, 0:3, 0:3, 0], np.zeros((3, 3))), "Rows/Columns of padding must be all zeros"
    assert np.allclose(x_pad[0, 3:6, 3:6, 0], x[0, :, :, 0]), "Center of padded array must equal original array"

    print("\033[92mAll tests passed!")


def conv_single_step_test(target):
    np.random.seed(3)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = target(a_slice_prev, W, b)

    assert type(Z) == np.float64, "You must cast the output to numpy float 64"

    print("\033[92mAll tests passed!")


def conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3):
    assert np.isclose(z_mean, 0.5511276474566768), f"Wrong value for z_mean, got {z_mean}"
    assert np.allclose(z_0_2_1, [-2.17796037, 8.07171329, -0.5772704,
                                  3.36286738, 4.48113645, -2.89198428,
                                  10.99288867, 3.03171932]), "Wrong values for z[0, 2, 1]"
    assert np.allclose(cache_0_1_2_3, [-1.1191154, 1.9560789, -0.3264995, -1.34267579]), "Wrong values for cache"

    print("\033[92mFirst Test: All tests passed!")


def conv_forward_test_2(target):
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)

    hparameters = {"pad": 1, "stride": 2}

    Z, cache_conv = target(A_prev, W, b, hparameters)

    assert type(Z) == np.ndarray, "Output must be a np.ndarray"
    assert Z.shape == (2, 3, 4, 8), f"Wrong shape: {Z.shape} != (2, 3, 4, 8)"
    assert type(cache_conv) == tuple, "Cache must be a tuple"
    assert len(cache_conv) == 4, "Cache must have 4 elements"

    print("\033[92mSecond Test: All tests passed!")


def pool_forward_test_1(target):
    np.random.seed(1)
    A_prev_case_1 = np.random.randn(2, 5, 5, 3)
    hparameters_case_1 = {"stride": 1, "f": 3}

    A, cache = target(A_prev_case_1, hparameters_case_1, mode="max")

    assert A.shape == (2, 3, 3, 3), f"Wrong shape for max pooling: {A.shape} != (2, 3, 3, 3)"
    assert np.allclose(A[1, 1], [[1.96710175, 0.84616065, 1.27375593],
                                  [1.96710175, 0.84616065, 1.23616403],
                                  [1.62765075, 1.12141771, 1.2245077]]), "Wrong values for max pooling"

    A, cache = target(A_prev_case_1, hparameters_case_1, mode="average")

    assert A.shape == (2, 3, 3, 3), f"Wrong shape for average pooling: {A.shape} != (2, 3, 3, 3)"

    print("\033[92mFirst Test: All tests passed!")


def pool_forward_test_2(target):
    np.random.seed(1)
    A_prev_case_2 = np.random.randn(2, 5, 5, 3)
    hparameters_case_2 = {"stride": 2, "f": 3}

    A, cache = target(A_prev_case_2, hparameters_case_2, mode="max")

    assert A.shape == (2, 2, 2, 3), f"Wrong shape for max pooling: {A.shape} != (2, 2, 2, 3)"

    A, cache = target(A_prev_case_2, hparameters_case_2, mode="average")

    assert A.shape == (2, 2, 2, 3), f"Wrong shape for average pooling: {A.shape} != (2, 2, 2, 3)"

    print("\033[92mSecond Test: All tests passed!")
