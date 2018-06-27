import numpy as np
import pandas as pd

from pytest import approx

import lenskit.metrics.topn as lm


def test_precision_empty_none():
    prec = lm.precision([], [1, 3])
    assert prec is None


def test_precision_simple_cases():
    prec = lm.precision([1, 3], [1, 3])
    assert prec == approx(1.0)

    prec = lm.precision([1], [1, 3])
    assert prec == approx(1.0)

    prec = lm.precision([1, 2, 3, 4], [1, 3])
    assert prec == approx(0.5)

    prec = lm.precision([1, 2, 3, 4], [1, 3, 5])
    assert prec == approx(0.5)

    prec = lm.precision([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = lm.precision([1, 2, 3, 4], range(4, 10))
    assert prec == approx(0.25)


def test_precision_series():
    prec = lm.precision(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), pd.Series([1, 3, 5]))
    assert prec == approx(0.5)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 10)))
    assert prec == approx(0.25)


def test_precision_series_set():
    prec = lm.precision(pd.Series([1, 2, 3, 4]), [1, 3, 5])
    assert prec == approx(0.5)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), range(4, 10))
    assert prec == approx(0.25)


def test_precision_series_index():
    prec = lm.precision(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5]))
    assert prec == approx(0.5)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 10)))
    assert prec == approx(0.25)


def test_precision_series_array():
    prec = lm.precision(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5]))
    assert prec == approx(0.5)

    prec = lm.precision(pd.Series([1, 2, 3, 4]), np.arange(4, 10, 1, 'u4'))
    assert prec == approx(0.25)


def test_precision_array():
    prec = lm.precision(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = lm.precision(np.array([1, 2, 3, 4]), np.array([1, 3, 5]))
    assert prec == approx(0.5)

    prec = lm.precision(np.array([1, 2, 3, 4]), np.arange(4, 10, 1, 'u4'))
    assert prec == approx(0.25)
