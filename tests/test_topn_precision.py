import numpy as np
import pandas as pd

from pytest import approx

from lenskit.topn import precision


def _test_prec(items, rel):
    recs = pd.DataFrame({'item': items})
    truth = pd.DataFrame({'item': rel}).set_index('item')
    return precision(recs, truth)


def test_precision_empty_none():
    prec = _test_prec([], [1, 3])
    assert prec is None


def test_precision_simple_cases():
    prec = _test_prec([1, 3], [1, 3])
    assert prec == approx(1.0)

    prec = _test_prec([1], [1, 3])
    assert prec == approx(1.0)

    prec = _test_prec([1, 2, 3, 4], [1, 3])
    assert prec == approx(0.5)

    prec = _test_prec([1, 2, 3, 4], [1, 3, 5])
    assert prec == approx(0.5)

    prec = _test_prec([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = _test_prec([1, 2, 3, 4], range(4, 10))
    assert prec == approx(0.25)


def test_precision_series():
    prec = _test_prec(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Series([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 10)))
    assert prec == approx(0.25)


def test_precision_series_set():
    prec = _test_prec(pd.Series([1, 2, 3, 4]), [1, 3, 5])
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), range(4, 10))
    assert prec == approx(0.25)


def test_precision_series_index():
    prec = _test_prec(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 10)))
    assert prec == approx(0.25)


def test_precision_series_array():
    prec = _test_prec(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(pd.Series([1, 2, 3, 4]), np.arange(4, 10, 1, 'u4'))
    assert prec == approx(0.25)


def test_precision_array():
    prec = _test_prec(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = _test_prec(np.array([1, 2, 3, 4]), np.array([1, 3, 5]))
    assert prec == approx(0.5)

    prec = _test_prec(np.array([1, 2, 3, 4]), np.arange(4, 10, 1, 'u4'))
    assert prec == approx(0.25)
