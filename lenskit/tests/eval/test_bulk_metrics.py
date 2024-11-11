import pandas as pd  # noqa: 401

from pytest import approx

from lenskit.metrics.basic import ListLength
from lenskit.metrics.bulk import RunAnalysis
from lenskit.metrics.predict import RMSE
from lenskit.metrics.ranking import NDCG, Precision
from lenskit.util.test import demo_recs, ml_ratings


def test_bulk_measure_function(ml_ratings: pd.DataFrame):
    bms = RunAnalysis()
    bms.add_metric(ListLength(), "length")
    bms.add_metric(RMSE)

    metrics = bms.compute(ml_ratings.rename(columns={"rating": "score"}), ml_ratings)
    stats = metrics.summary()
    assert stats.loc["length", "mean"] == approx(ml_ratings["user"].value_counts().mean())
    assert stats.loc["RMSE", "mean"] == approx(0)


def test_recs(demo_recs):
    train, test, recs = demo_recs

    bms = RunAnalysis()
    bms.add_metric(ListLength())
    bms.add_metric(Precision())
    bms.add_metric(NDCG())

    metrics = bms.compute(recs, test)
    scores = metrics.list_scores()
    stats = metrics.summary()
    print(stats)
    for m in bms.metrics:
        assert stats.loc[m.label, "mean"] == approx(scores[m.label].mean())
