import numpy as np
import pandas as pd

from pytest import approx

import lenskit.metrics.topn as lm


def test_recall_empty_zero():
    prec = lm.recall([], set([1, 3]))
    assert prec == approx(0)


def test_recall_norel_na():
    prec = lm.recall([1, 3], set())
    assert np.isnan(prec)


def test_recall_simple_cases():
    prec = lm.recall([1, 3], set([1, 3]))
    assert prec == approx(1.0)

    prec = lm.recall([1], set([1, 3]))
    assert prec == approx(0.5)

    prec = lm.recall([1, 2, 3, 4], set([1, 3]))
    assert prec == approx(1.0)

    prec = lm.recall([1, 2, 3, 4], set([1, 3, 5]))
    assert prec == approx(2.0 / 3)

    prec = lm.recall([1, 2, 3, 4], range(5, 10))
    assert prec == approx(0.0)

    prec = lm.recall([1, 2, 3, 4], range(4, 9))
    assert prec == approx(0.2)


def test_recall_series():
    prec = lm.recall(pd.Series([1, 3]), pd.Series([1, 3]))
    assert prec == approx(1.0)

    prec = lm.recall(pd.Series([1, 2, 3]), pd.Series([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = lm.recall(pd.Series([1, 2, 3, 4]), pd.Series(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_set():
    prec = lm.recall(pd.Series([1, 2, 3, 4]), set([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = lm.recall(pd.Series([1, 2, 3, 4]), range(4, 9))
    assert prec == approx(0.2)


def test_recall_series_index():
    prec = lm.recall(pd.Series([1, 3]), pd.Index([1, 3]))
    assert prec == approx(1.0)

    prec = lm.recall(pd.Series([1, 2, 3, 4]), pd.Index([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = lm.recall(pd.Series([1, 2, 3, 4]), pd.Index(range(4, 9)))
    assert prec == approx(0.2)


def test_recall_series_array():
    prec = lm.recall(pd.Series([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = lm.recall(pd.Series([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = lm.recall(pd.Series([1, 2, 3, 4]), np.arange(4, 9, 1, 'u4'))
    assert prec == approx(0.2)


def test_recall_array():
    prec = lm.recall(np.array([1, 3]), np.array([1, 3]))
    assert prec == approx(1.0)

    prec = lm.recall(np.array([1, 2, 3, 4]), np.array([1, 3, 5, 7]))
    assert prec == approx(0.5)

    prec = lm.recall(np.array([1, 2, 3, 4]), np.arange(4, 9, 1, 'u4'))
    assert prec == approx(0.2)
