# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd

from pytest import approx, mark, raises

from lenskit.basic import PopScorer
from lenskit.data import ItemList
from lenskit.metrics import NDCG, Recall
from lenskit.metrics._accum import MetricAccumulator
from lenskit.metrics.basic import ListLength
from lenskit.metrics.predict import RMSE
from lenskit.splitting import split_temporal_fraction

_log = logging.getLogger(__name__)


def test_accumulator_initial_state():
    acc = MetricAccumulator()
    assert acc.metrics == []
    assert acc.list_metrics().empty
    assert acc.summary_metrics().empty


def test_unmeasured_metrics():
    acc = MetricAccumulator()
    acc.add_metric(Recall(5))
    acc.add_metric(NDCG(5))

    df = acc.list_metrics()
    summary = acc.summary_metrics()

    assert df.empty
    assert approx(summary.loc["Recall@5", "mean"]) == 0.0
    assert approx(summary.loc["NDCG@5", "mean"]) == 0.0


def test_basic_metric_flow():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([1, 2, 3], user=["u1"] * 3), ItemList([2, 3]), user="u1")
    acc.measure_list(ItemList([4, 5], user=["u2"] * 2), ItemList([5]), user="u2")

    list_metrics = acc.list_metrics()
    summary = acc.summary_metrics()

    assert isinstance(list_metrics, pd.DataFrame)
    assert len(list_metrics) == 2
    assert "N" in list_metrics.columns
    assert set(list_metrics["N"]) == {2.0, 3.0}

    assert isinstance(summary, pd.DataFrame)
    assert "mean" in summary.columns
    assert approx(summary.loc["N", "mean"]) == 2.5


def test_metric_accum(ml_ds):
    split = split_temporal_fraction(ml_ds, 0.2, filter_test_users=True)

    scorer = PopScorer()
    scorer.train(split.train)

    acc = MetricAccumulator()
    acc.add_metric(Recall(10))
    acc.add_metric(NDCG(10))

    all_train_items = split.train.items.ids()

    for user, truth_il in split.test:
        scores = scorer(ItemList(all_train_items, user=[user.user_id] * len(all_train_items)))
        top_10 = scores.top_n(10)

        recs_il = ItemList(top_10, user=[user.user_id] * len(top_10), ordered=True)
        acc.measure_list(recs_il, truth_il, user=user.user_id)

    list_metrics = acc.list_metrics()
    summary = acc.summary_metrics()

    # per-list metrics
    assert isinstance(list_metrics, pd.DataFrame)
    assert len(list_metrics) == len(split.test)
    for metric in ["Recall@10", "NDCG@10"]:
        assert metric in list_metrics.columns
        assert np.all(list_metrics[metric] >= 0)
        assert np.all(list_metrics[metric] <= 1)

    # summary metrics
    assert isinstance(summary, pd.DataFrame)
    for metric in ["Recall@10", "NDCG@10"]:
        assert metric in summary.index
        val = summary.loc[metric]
        assert 0 <= val["mean"] <= 1


def test_to_result_conversion():
    acc = MetricAccumulator()
    acc.add_metric(Recall(5))
    acc.add_metric(NDCG(5))

    recs1 = ItemList([1, 2, 3], user=["u1"] * 3, ordered=True)
    truth1 = ItemList([2, 3])
    recs2 = ItemList([4, 5], user=["u2"] * 2, ordered=True)
    truth2 = ItemList([5])

    acc.measure_list(recs1, truth1, user="u1")
    acc.measure_list(recs2, truth2, user="u2")

    result = acc.to_result()

    list_results = result.list_metrics()
    assert isinstance(list_results, pd.DataFrame)

    g = result.global_metrics()
    assert isinstance(g, pd.Series)

    summary_df = result.list_summary()
    assert isinstance(summary_df, pd.DataFrame)

    for metric_name in ["Recall@5", "NDCG@5"]:
        assert metric_name in list_results.columns
        assert metric_name in g.index
        assert metric_name in summary_df.index

        assert 0.0 <= g[metric_name] <= 1.0
        assert 0.0 <= summary_df.loc[metric_name, "mean"] <= 1.0


def test_measure_list_multiple_key_fields():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([1, 2, 3], user=["u1"] * 3), ItemList([2]), user="u1", fold=1)
    acc.measure_list(ItemList([4, 5], user=["u2"] * 2), ItemList([5]), user="u2", fold=1)

    assert acc._list_keys == [("u1", 1), ("u2", 1)]
    assert acc._key_fields == ["user", "fold"]

    metrics = acc.list_metrics()
    assert len(metrics) == 2
    assert set(metrics.index.names) == {"user", "fold"}

    summary = acc.summary_metrics()
    assert "mean" in summary.columns


def test_measure_list_keys_work_as_expected():
    acc = MetricAccumulator()
    acc.add_metric(Recall(2))

    acc.measure_list(ItemList([1, 2], user=["u1"] * 2), ItemList([2]), user="u1", session="a")
    acc.measure_list(ItemList([3, 4], user=["u1"] * 2), ItemList([3]), user="u1", session="b")

    metrics = acc.list_metrics()
    assert len(metrics) == 2
    assert ("u1", "a") in metrics.index
    assert ("u1", "b") in metrics.index


def test_summary_defaults_to_mean():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([1, 2], user=["u1"] * 2), ItemList([2]), user="u1")
    acc.measure_list(ItemList([3, 4, 5], user=["u2"] * 3), ItemList([4]), user="u2")

    summary = acc.summary_metrics()
    assert approx(summary.loc["N", "mean"]) == 2.5


def test_duplicate_metric_labels_raise():
    acc = MetricAccumulator()
    acc.add_metric(ListLength, label="dup")
    acc.add_metric(ListLength, label="dup")
    with raises(RuntimeError):
        acc._validate_setup()


def test_metric_measure_list_exception_logs_warning(caplog):
    acc = MetricAccumulator()

    class BadMetric:
        pass

    acc.add_metric(BadMetric())

    with caplog.at_level(logging.WARNING):
        acc.measure_list(ItemList([1], user=[1]), ItemList([1]), user=1)

    assert any("Error computing metric" in r.message for r in caplog.records)


def test_measure_list_mixed_result_types():
    acc = MetricAccumulator()

    class DummyMetric(ListLength):
        label = "D"

        def measure_list(self, recs, test):
            if len(acc._list_data.get(self.label, [])) % 2 == 0:  # returns scaler
                return 1.0
            else:
                return {"b": 2.0}  # returns dict

        def summarize(self, values):
            return {"mean": np.mean([v if isinstance(v, float) else v["a"] for v in values])}

    acc.add_metric(DummyMetric())

    acc.measure_list(ItemList([1], user=["u1"]), ItemList([1]), user="u1")
    acc.measure_list(ItemList([2], user=["u2"]), ItemList([2]), user="u2")

    df = acc.list_metrics()
    assert "D.b" in df.columns
