import pandas as pd
import numpy as np

import lenskit.util.test as lktu
from lenskit.algorithms.basic import PopScore
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.ranking import PlackettLuce


def test_plackett_luce_rec():
    pop = PopScore()
    algo = PlackettLuce(pop, rng_spec='user')
    algo.fit(lktu.ml_test.ratings)

    items = lktu.ml_test.ratings['item'].unique()
    nitems = len(items)

    recs1 = algo.recommend(2038, 100)
    recs2 = algo.recommend(2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100

    # we don't get exactly the same set of recs
    assert set(recs1['item']) != set(recs2['item'])

    recs_all = algo.recommend(2038)
    assert len(recs_all) == nitems
    assert set(items) == set(recs_all['item'])


def test_plackett_luce_pred():
    bias = Bias()
    algo = PlackettLuce(bias, rng_spec='user')
    algo.fit(lktu.ml_test.ratings)

    items = lktu.ml_test.ratings['item'].unique()
    nitems = len(items)

    recs1 = algo.recommend(2038, 100)
    recs2 = algo.recommend(2028, 100)
    assert len(recs1) == 100
    assert len(recs2) == 100

    # we don't get exactly the same set of recs
    assert set(recs1['item']) != set(recs2['item'])

    recs_all = algo.recommend(2038)
    assert len(recs_all) == nitems
    assert set(items) == set(recs_all['item'])
