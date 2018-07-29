from lenskit import topn

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
