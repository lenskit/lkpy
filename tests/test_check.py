from pytest import raises

import numpy as np

from lenskit.check import check_value, check_dimension


def test_check_value_passes():
    check_value(True, "value should be true")
    # it should complete successfully!


def test_check_value_fails():
    with raises(ValueError):
        check_value(False, "value should be true")


def test_check_meaningful_value_fails():
    with raises(ValueError):
        check_value(5 < 4, "five should be less than four")


def test_check_meaningful_value_succeeds():
    check_value(3 < 4, "three should be less than four")
    # it should complete successfully


def test_check_dim_len():
    check_dimension([], [])
    with raises(ValueError):
        check_dimension([], [3])
    with raises(ValueError):
        check_dimension([1], [])
    check_dimension(range(10), range(10, 20))


def test_check_dim_shape():
    check_dimension(np.arange(5), np.arange(5), d1=0, d2=0)
    with raises(ValueError):
        check_dimension(np.arange(5), np.arange(6), d1=0, d2=0)
    with raises(ValueError):
        check_dimension(np.random.randn(8, 10),
                        np.random.randn(8, 9), d1=1, d2=1)

    check_dimension(np.random.randn(8, 10),
                    np.random.randn(23, 8), d1=0, d2=1)
