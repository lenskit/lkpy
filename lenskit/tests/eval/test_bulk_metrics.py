import pandas as pd  # noqa: 401

from pytest import approx

from lenskit.data import ItemListCollection
from lenskit.data.schemas import ITEM_COMPAT_COLUMN, USER_COMPAT_COLUMN
from lenskit.metrics.basic import ListLength
from lenskit.metrics.bulk import RunAnalysis
from lenskit.metrics.predict import RMSE
from lenskit.metrics.ranking import NDCG, Precision
from lenskit.util.test import demo_recs, ml_ratings


def test_bulk_measure_function(ml_ratings: pd.DataFrame):
    bms = RunAnalysis()
    bms.add_metric(ListLength(), "length")
    bms.add_metric(RMSE)

    data = ItemListCollection.from_df(
        ml_ratings.rename(columns={"rating": "score"}), USER_COMPAT_COLUMN, ITEM_COMPAT_COLUMN
    )
    truth = ItemListCollection.from_df(ml_ratings, USER_COMPAT_COLUMN, ITEM_COMPAT_COLUMN)

    metrics = bms.compute(data, truth)
    stats = metrics.summary()
    assert stats.loc["length", "mean"] == approx(ml_ratings["user"].value_counts().mean())
    assert stats.loc["RMSE", "mean"] == approx(0)


def test_recs(demo_recs):
    train, test, recs = demo_recs

    bms = RunAnalysis()
    bms.add_metric(ListLength())
    bms.add_metric(Precision())
    bms.add_metric(NDCG())

    recs = ItemListCollection.from_df(recs, USER_COMPAT_COLUMN, ITEM_COMPAT_COLUMN)
    test = ItemListCollection.from_df(test, USER_COMPAT_COLUMN, ITEM_COMPAT_COLUMN)
    metrics = bms.compute(recs, test)
    scores = metrics.list_scores()
    stats = metrics.summary()
    print(stats)
    for m in bms.metrics:
        assert stats.loc[m.label, "mean"] == approx(scores[m.label].mean())
