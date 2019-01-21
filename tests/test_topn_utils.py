from lenskit import topn
from lenskit.algorithms import CandidateSelector

import pandas as pd
import numpy as np

import lk_test_utils as lktu


def test_unrated():
    ratings = lktu.ml_pandas.renamed.ratings
    unrate = topn.UnratedCandidates(ratings)

    cs = unrate(100)
    items = ratings.item.unique()
    rated = ratings[ratings.user == 100].item.unique()
    assert len(cs) == len(items) - len(rated)
    assert len(np.intersect1d(cs, rated)) == 0


def test_cs_rated_items_series():
    "rated_items should de-index series"
    items = ['a', 'b', 'wombat']
    series = pd.Series(np.random.randn(3), index=items)

    i2 = CandidateSelector.rated_items(series)
    assert isinstance(i2, np.ndarray)
    assert all(i2 == items)


def test_cs_rated_items():
    "rated_items should return list as array"
    items = ['a', 'b', 'wombat']

    i2 = CandidateSelector.rated_items(items)
    assert isinstance(i2, np.ndarray)
    assert all(i2 == items)


def test_cs_rated_items_array():
    "rated_items should return array as itself"
    items = ['a', 'b', 'wombat']
    items = np.array(items)

    i2 = CandidateSelector.rated_items(items)
    assert isinstance(i2, np.ndarray)
    assert all(i2 == items)
    assert i2 is items
