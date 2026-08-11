import numpy as np

from pytest import fixture

from lenskit import Pipeline, predict, recommend, score, topn_pipeline
from lenskit.data import Dataset
from lenskit.knn import ItemKNNScorer


@fixture(scope="module")
def pipeline(ml_ds: Dataset):
    pipe = topn_pipeline(ItemKNNScorer(max_nbrs=20), predicts_ratings=True)
    pipe.train(ml_ds)
    yield pipe


def test_simple_recommend(pipeline: Pipeline):
    recs = recommend(pipeline, 10, n=10)
    assert len(recs) == 10


def test_simple_recommend_items(ml_ds: Dataset, pipeline: Pipeline):
    items = ml_ds.items.ids()
    recs = recommend(pipeline, 10, n=10, items=items)
    assert len(recs) == 10


def test_simple_score(ml_ds: Dataset, pipeline: Pipeline, rng: np.random.Generator):
    items = rng.choice(ml_ds.items.ids(), 50, replace=False)
    preds = score(pipeline, 10, items)
    assert len(preds) == 50
    scores = preds.scores()
    assert scores is not None
    assert np.any(np.isfinite(scores))


def test_simple_predict(ml_ds: Dataset, pipeline: Pipeline, rng: np.random.Generator):
    items = rng.choice(ml_ds.items.ids(), 50, replace=False)
    preds = predict(pipeline, 10, items)
    assert len(preds) == 50
    scores = preds.scores()
    assert scores is not None
    assert np.all(np.isfinite(scores))
