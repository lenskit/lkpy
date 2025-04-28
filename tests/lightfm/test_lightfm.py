import logging
import pickle

import numpy as np
import pandas as pd
from pytest import approx, mark
from lenskit.data import ItemList ,from_interactions_df
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests
from lenskit.Lightfm import LightFMScorer
_log = logging.getLogger(__name__)

# Sample dataset for testing
simple_df = pd.DataFrame(
    {
        "item": [1, 1, 2, 3],
        "user": [10, 12, 10, 13],
        "rating": [4.0, 3.0, 5.0, 2.0]
    }
)
simple_ds = from_interactions_df(simple_df)


class TestLightFM(BasicComponentTests, ScorerTests):
    component = LightFMScorer


def test_lightfm_basic_train():
    algo = LightFMScorer()
    algo.train(simple_ds)
    assert algo.model is not None


def test_lightfm_predict_basic():
    _log.info("Testing LightFM scoring")
    algo = LightFMScorer()
    algo.train(simple_ds)
    
    preds = algo(10, ItemList([3]))
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert preds.index[0] == 3
    assert preds.loc[3] >= 0 


def test_lightfm_predict_unknown_user():
    algo = LightFMScorer()
    algo.train(simple_ds)

    preds = algo(50, ItemList([3]))  # Unknown user
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.isnan(preds.loc[3])


def test_lightfm_predict_unknown_item():
    algo = LightFMScorer()
    algo.train(simple_ds)

    preds = algo(10, ItemList([99]))  # Unknown item
    assert len(preds) == 1
    preds = preds.scores("pandas", index="ids")
    assert preds is not None
    assert np.isnan(preds.loc[99])


@mark.slow
def test_lightfm_serialization():
    algo = LightFMScorer()
    algo.train(simple_ds)

    _log.info("Serializing LightFM model")
    mod = pickle.dumps(algo)
    _log.info("Serialized to %d bytes", len(mod))
    
    new_algo = pickle.loads(mod)
    assert new_algo is not algo  
    assert new_algo.model is not None  
    
    preds1 = algo(10, ItemList([3])).scores("pandas", index="ids")
    preds2 = new_algo(10, ItemList([3])).scores("pandas", index="ids")
    assert np.allclose(preds1, preds2, equal_nan=True)


@mark.slow
@mark.eval
def test_lightfm_batch_accuracy(ml_100k: pd.DataFrame):
    data = from_interactions_df(ml_100k)
    lightfm_algo = LightFMScorer()
    results = quick_measure_model(lightfm_algo, data, metric="ndcg")

    assert results.list_summary().loc["NDCG", "mean"] == approx(0.1, abs=0.05)