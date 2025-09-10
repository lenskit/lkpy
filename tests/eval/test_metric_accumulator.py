# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx, mark, raises

from lenskit.basic import PopScorer
from lenskit.data import ItemList, ItemListCollection
from lenskit.metrics import NDCG, Recall
from lenskit.metrics._accum import MetricAccumulator, MetricWrapper
from lenskit.metrics._base import Metric
from lenskit.metrics.basic import ListLength
from lenskit.splitting import split_temporal_fraction

_log = logging.getLogger(__name__)


def test_accumulator_initial_state():
    """Test MetricAccumulator has empty state."""
    acc = MetricAccumulator()
    assert acc.metrics == []
    assert acc.list_metrics().empty
    assert acc.summary_metrics().empty


def test_unmeasured_metrics():
    """Test that unmeasured metrics return default values in summary."""
    acc = MetricAccumulator()
    acc.add_metric(Recall(5))
    acc.add_metric(NDCG(5))

    df = acc.list_metrics()
    summary = acc.summary_metrics()

    assert df.empty
    assert summary.loc["Recall@5", "mean"] == 0.0
    assert summary.loc["NDCG@5", "mean"] == 0.0


def test_basic_metric_flow():
    """Test basic metric measurement and aggregation flow."""
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
    """Test full metric accumulation workflow with real dataset."""
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


def test_measure_list_multiple_key_fields():
    """Test measuring lists with multiple identifying key fields."""
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([1, 2, 3], user=["u1"] * 3), ItemList([2]), user="u1", fold=1)
    acc.measure_list(ItemList([4, 5], user=["u2"] * 2), ItemList([5]), user="u2", fold=1)

    # check internal state properly
    assert acc._key_fields == ["user", "fold"]
    assert len(acc._records) == 2

    metrics = acc.list_metrics()
    assert len(metrics) == 2
    assert set(metrics.index.names) == {"user", "fold"}

    summary = acc.summary_metrics()
    assert "mean" in summary.columns


def test_measure_list_keys_work_as_expected():
    acc = MetricAccumulator()
    acc.add_metric(Recall(2))

    acc.measure_list(
        ItemList([1, 2], user=["u1"] * 2, ordered=True), ItemList([2]), user="u1", session="a"
    )
    acc.measure_list(
        ItemList([3, 4], user=["u1"] * 2, ordered=True), ItemList([3]), user="u1", session="b"
    )

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


def test_measure_list_mixed_result_types():
    """Test handling of metrics that return both scalar and dictionary results."""
    acc = MetricAccumulator()

    class DummyMetric(Metric):
        label = "D"
        default = 0.0

        def __init__(self):
            self.call_count = 0

        def measure_list(self, recs, test):
            self.call_count += 1
            if self.call_count % 2 == 1:  # returns scalar
                return 1.0
            else:
                return {"b": 2.0}  # returns dict

        def summarize(self, values):
            return {"mean": np.mean([v if isinstance(v, float) else v["b"] for v in values])}

    acc.add_metric(DummyMetric())

    acc.measure_list(ItemList([1], user=["u1"]), ItemList([1]), user="u1")
    acc.measure_list(ItemList([2], user=["u2"]), ItemList([2]), user="u2")

    df = acc.list_metrics()
    assert "D.b" in df.columns


def test_add_metric_with_class_type():
    """Test adding metric by passing class type instead of instance."""
    acc = MetricAccumulator()

    acc.add_metric(ListLength)

    assert len(acc.metrics) == 1
    assert isinstance(acc.metrics[0].metric, ListLength)
    assert acc.metrics[0].label == "N"


def test_custom_label_override():
    """Test that custom labels override default metric labels."""
    acc = MetricAccumulator()

    # custom label should override default
    acc.add_metric(ListLength(), label="CustomLength")
    assert acc.metrics[0].label == "CustomLength"


def test_complex_summary():
    """Test complex nested summary data structures."""

    class ComplexMetric(Metric):
        label = "Complex"
        default = 0.0

        def measure_list(self, recs, test):
            return {"sub1": 1.0, "sub2": 2.0}

        def summarize(self, values):
            return {"sub1": {"mean": 1.5, "std": 0.5}, "sub2": {"mean": 2.5, "std": 0.7}}

    acc = MetricAccumulator()
    acc.add_metric(ComplexMetric())

    acc.measure_list(ItemList([1], user=["u1"]), ItemList([1]), user="u1")

    summary = acc.summary_metrics()
    assert len(summary) > 0


@mark.parametrize(
    "values,expected",
    [
        ([], {"mean": None, "median": None, "std": None}),
        ([None, None], {"mean": None, "median": None, "std": None}),
        ([42], {"mean": 42.0, "median": 42.0, "std": 0.0}),
        ([1, 2, 3, 4], None),
        (pa.array([1, 2, 3]), None),
        (pa.chunked_array([[1, 2], [3]]), None),
    ],
)
def test_default_summarize(values, expected):
    """Test default summarization behavior with various input types."""
    mw = MetricWrapper(metric=None, label="test")
    result = mw._default_summarize(values)

    if expected is not None:
        assert result == expected
    else:
        vals = [1, 2, 3, 4] if isinstance(values, list) else [1, 2, 3]
        assert result["mean"] == approx(np.mean(vals))
        assert result["median"] == approx(np.median(vals))
        assert result["std"] == approx(np.std(vals, ddof=1))


def test_default_value_handling():
    """Test different default value scenarios and inheritance."""

    # ListMetric with default
    class MetricWithDefault(ListLength):
        label = "WithDefault"
        default = 42.0

        def measure_list(self, recs, test):
            return 1.0

    acc = MetricAccumulator()
    acc.add_metric(MetricWithDefault())
    assert acc.metrics[0].default == 42.0

    # non-ListMetric (should default to 0.0)
    def simple_metric(recs, test):
        return 1.0

    acc = MetricAccumulator()
    acc.add_metric(simple_metric)
    assert acc.metrics[0].default == 0.0

    # custom default override
    acc = MetricAccumulator()
    acc.add_metric(MetricWithDefault(), default=99.0)
    assert acc.metrics[0].default == 99.0
