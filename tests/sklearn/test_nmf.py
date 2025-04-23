import logging
import pickle

import numpy as np
import pandas as pd

from pytest import approx, importorskip, mark

from lenskit.data import Dataset, ItemList, from_interactions_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests

nmf = importorskip("lenskit.sklearn.nmf")

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame(
    {"item": [1, 1, 2, 3], "user": [10, 12, 10, 13], "rating": [4.0, 3.0, 5.0, 2.0]}
)
simple_ds = from_interactions_df(simple_df)

class TestNMF(BasicComponentTests, ScorerTests):
    component = nmf.NMFScorer

def test_nmf_basic_build():
    algo = nmf.NMFScorer(features=2)
    algo.train(simple_ds)

    assert algo.user_components_.shape == (3, 3)

def test_nmf_predict_basic():
    _log.info("NMF input data:\n%s", simple_df)
    algo = nmf.NMFScorer(features=2)
    algo.train(simple_ds)
    _log.info("user matrix:\n%s", str(algo.user_components_))
    _log.info("item matrix:\n%s", str(algo.item_components_))

    preds = algo(10, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0
    assert preds.loc[3] <= 5

def test_nmf_predict_bad_item():
    algo = nmf.NMFScorer(features=2)
    algo.train(simple_ds)

    preds = algo(10, ItemList([4]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 4
    assert np.isnan(preds.loc[4])

def test_nmf_predict_bad_user():
    algo = nmf.NMFScorer(features=2)
    algo.train(simple_ds)

    preds = algo(50, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert np.isnan(preds.loc[3])

@mark.slow
def test_nmf_save_load(ml_ds: Dataset):
    original = nmf.NMFScorer(features=20)
    original.train(ml_ds)

    mod = pickle.dumps(original)
    _log.info("serialized to %d bytes", len(mod))
    algo = pickle.loads(mod)

    assert algo.bias_.global_bias == original.bias_.global_bias
    assert np.all(algo.bias_.user_biases == original.bias_.user_biases)
    assert np.all(algo.bias_.item_biases == original.bias_.item_biases)
    assert np.all(algo.user_components_ == original.user_components_)

@mark.slow
@mark.eval
def test_nmf_batch_accuracy(rng, ml_100k: pd.DataFrame):
    data = from_interactions_df(ml_100k)
    svd_algo = nmf.NMFScorer(features=25)
    results = quick_measure_model(svd_algo, data, predicts_ratings=True, rng=rng)

    ndcg = results.list_summary().loc["NDCG", "mean"]
    _log.info("nDCG for users is %.4f", ndcg)
    assert ndcg > 0.22