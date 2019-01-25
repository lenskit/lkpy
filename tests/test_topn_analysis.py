import numpy as np
import pandas as pd

from pytest import approx

from lenskit import topn
import lk_test_utils as lktu


def test_run_one():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)

    recs = pd.DataFrame({'user': 1, 'item': [5]})
    truth = pd.DataFrame({'user': 1, 'item': [1, 2, 3], 'rating': [3.0, 5.0, 4.0]})

    res = rla.compute(recs, truth)

    assert len(res) == 1
    assert all(res.user == 1)
    assert all(res.precision == 0.0)
    assert all(res.recall.isna())


def test_run_two():
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.precision)
    rla.add_metric(topn.recall)
    rla.add_metric(topn.ndcg)

    recs = pd.DataFrame({
        'data': 'a',
        'user': ['a', 'a', 'a', 'b', 'b'],
        'item': [2, 3, 1, 4, 5]
    })
    truth = pd.DataFrame({
        'user': ['a', 'a', 'a', 'b', 'b', 'b'],
        'item': [1, 2, 3, 1, 5, 6],
        'rating': [3.0, 5.0, 4.0, 3.0, 5.0, 4.0]
    })

    res = rla.compute(recs, truth)
    assert len(res) == 2
    assert all(res.user == ['a', 'b'])
    assert res.ndcg == approx([1.0, 0.0])
    assert res.precision == approx(1.0, 1/2)
    assert res.precision == approx(1.0, 1/3)
