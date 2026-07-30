from lenskit.metrics import MeasurementCollector, RunAnalysis, UniqueItemCount
from lenskit.testing import demo_recs, ml_ratings


def test_recs(demo_recs):
    split, recs = demo_recs

    mc = MeasurementCollector()
    mc.add_metric(UniqueItemCount())

    metrics = mc.measure_run(recs, split.test)
    scores = metrics.summary_metrics
    assert "UniqueItemCount" in scores

    items = {i for il in recs.lists() for i in il.ids()}
    assert scores["UniqueItemCount"] == len(items)
