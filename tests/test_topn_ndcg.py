import numpy as np
import pandas as pd

from pytest import approx

import lenskit.metrics.topn as lm


def test_dcg_empty():
    "empty should be zero"
    assert lm._dcg(np.array([])) == approx(0)


def test_dcg_zeros():
    assert lm._dcg(np.zeros(10)) == approx(0)


def test_dcg_single():
    "a single element should be scored at the right place"
    assert lm._dcg(np.array([0.5])) == approx(0.5)
    assert lm._dcg(np.array([0, 0.5])) == approx(0.5)
    assert lm._dcg(np.array([0, 0, 0.5])) == approx(0.5 / np.log2(3))
    assert lm._dcg(np.array([0, 0, 0.5, 0])) == approx(0.5 / np.log2(3))


def test_dcg_mult():
    "multiple elements should score correctly"
    assert lm._dcg(np.array([np.e, np.pi])) == approx(np.e + np.pi)
    assert lm._dcg(np.array([np.e, 0, 0, np.pi])) == approx(np.e + np.pi / np.log2(4))
