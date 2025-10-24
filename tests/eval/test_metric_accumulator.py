# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging

import numpy as np
import pandas as pd
import pyarrow as pa

from pytest import approx, fixture, mark, raises

from lenskit.basic import PopScorer
from lenskit.data import ItemList, ItemListCollection
from lenskit.metrics import NDCG, Recall
from lenskit.metrics._accum import MeasurementCollector, MetricWrapper
from lenskit.metrics._base import DecomposedMetric, GlobalMetric, ListMetric, Metric
from lenskit.metrics.basic import ListLength
from lenskit.splitting import split_temporal_fraction

_log = logging.getLogger(__name__)


@fixture
def basic_accumulator():
    acc = MeasurementCollector()
    acc.add_metric(Recall(5))
    acc.add_metric(NDCG(5))
    return acc


@fixture
def sample_lists():
    return {
        "recs1": ItemList([1, 2, 3], user=["u1"] * 3),
        "test1": ItemList([2, 3]),
        "recs2": ItemList([4, 5], user=["u2"] * 2),
        "test2": ItemList([5]),
    }


def test_accumulator_empty_and_unmeasured_defaults():
    acc = MeasurementCollector()
    # initial state
    assert acc.metrics == []
    assert acc.list_metrics().empty
    assert not acc.summary_metrics()

    # measuring with no metrics
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    assert acc.list_metrics().empty
    assert not acc.summary_metrics()

    # add metrics but do not measure
    acc.add_metric(Recall(5))
    acc.add_metric(NDCG(5))
    assert not acc.summary_metrics()


# measuring lists


def test_accumulator_measures_list_and_summary(sample_lists):
    acc = MeasurementCollector()
    acc.add_metric(ListLength())

    acc.measure_list(sample_lists["recs1"], sample_lists["test1"], user="u1")
    acc.measure_list(sample_lists["recs2"], sample_lists["test2"], user="u2")

    list_metrics = acc.list_metrics()
    summary = acc.summary_metrics()

    assert len(list_metrics) == 2
    assert set(list_metrics["N"]) == {2.0, 3.0}
    assert approx(summary["N"]) == 2.5


def test_accumulator_empty_itemlists():
    acc = MeasurementCollector()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([]), ItemList([1, 2]), user="u1")
    acc.measure_list(ItemList([1, 2]), ItemList([]), user="u2")
    acc.measure_list(ItemList([]), ItemList([]), user="u3")

    metrics = acc.list_metrics()
    assert len(metrics) == 3
    assert all(metrics["N"] >= 0)


def test_list_metrics_no_key_fields():
    acc = MeasurementCollector()
    acc.add_metric(ListLength())
    acc.measure_list(ItemList([1, 2]), ItemList([1]))
    metrics = acc.list_metrics()
    assert len(metrics) == 1
    assert metrics.index.names == [None]


# metric wrapping


def test_wrap_metric_with_class():
    acc = MeasurementCollector()
    wrapper = acc._wrap_metric(ListLength, label="ListLength", default=None)
    assert isinstance(wrapper.metric, ListLength)
    assert wrapper.label == "ListLength"


def test_wrap_metric_function_label():
    def custom_metric(recs, test):
        return len(recs)

    acc = MeasurementCollector()
    wrapper = acc._wrap_metric(custom_metric, None, None)
    assert wrapper.label == "function"


# key fields and duplicates


@mark.parametrize(
    "keys,expected_names",
    [
        ({"user": "u1"}, ["user"]),
        ({"user": "u1", "fold": 1}, ["user", "fold"]),
        ({"user": "u1", "session": "a", "fold": 1}, ["user", "session", "fold"]),
    ],
)
def test_accumulator_key_fields(keys, expected_names):
    acc = MeasurementCollector()
    acc.add_metric(ListLength())
    acc.measure_list(ItemList([1, 2]), ItemList([2]), **keys)
    assert acc._key_fields == expected_names
    metrics = acc.list_metrics()
    assert set(metrics.index.names) == set(expected_names)


# custom metrics and types


@mark.parametrize(
    "metric_input,expected_type",
    [
        (ListLength(), ListLength),
        (ListLength, ListLength),
        (lambda r, t: len(r), type(lambda: None)),
    ],
)
def test_add_metric_types(metric_input, expected_type):
    acc = MeasurementCollector()
    acc.add_metric(metric_input)
    assert isinstance(acc.metrics[0].metric, expected_type)


def test_custom_labels_and_defaults():
    acc = MeasurementCollector()
    acc.add_metric(ListLength(), label="CustomLength")
    assert acc.metrics[0].label == "CustomLength"
    acc.add_metric(ListLength(), default=99.0)
    assert acc.metrics[1].default == 99.0


# global, callable, scalar metrics


def test_global_metric():
    class DummyGlobalMetric(GlobalMetric):
        label = "dummy_global"

        def measure_run(self, run, test):
            return 42.0

    acc = MeasurementCollector()
    acc.add_metric(DummyGlobalMetric())
    result = acc.metrics[0].measure_run(ItemListCollection([]), ItemListCollection([]))
    assert result == 42.0
    assert acc.metrics[0].is_global
    assert not acc.metrics[0].is_listwise


def test_callable_metric():
    def _callable(recs, test):
        return len(recs)

    acc = MeasurementCollector()
    acc.add_metric(_callable, label="callable_metric")
    recs = ItemList([1, 2, 3])
    test_il = ItemList([1])
    result = acc.metrics[0].measure_list(recs, test_il)
    assert result == 3
    wrapper = acc.metrics[0]
    assert wrapper.is_listwise
    assert not wrapper.is_global


def test_scalar_metric():
    class ScalarMetric(Metric):
        label = "scalar"

        def measure_list(self, recs, test):
            return 5

        def summarize(self, values):
            return 99.0

    acc = MeasurementCollector()
    acc.add_metric(ScalarMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    summary = acc.summary_metrics()
    assert summary["scalar"] == 99.0


# metrics returning mixed


def test_mixed_metric_behavior():
    class MixedMetric(Metric):
        label = "Mixed"

        def __init__(self):
            self.call_count = 0

        def measure_list(self, recs, test):
            self.call_count += 1
            return 1.0 if self.call_count % 2 == 1 else {"b": 2.0}

        def extract_list_metrics(self, data):
            return data

        def summarize(self, values):
            numeric_vals = [v if isinstance(v, float) else v["b"] for v in values]
            return {"mean": np.mean(numeric_vals)}

    acc = MeasurementCollector()
    acc.add_metric(MixedMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    acc.measure_list(ItemList([2]), ItemList([2]), user="u2")

    df = acc.list_metrics()
    summary = acc.summary_metrics()

    assert "Mixed.mean" in summary
    assert summary["Mixed.mean"] == 1.5
    assert len(df) == 2


def test_none_extract_metric():
    class NoneExtractMetric(Metric):
        label = "none_extract"

        def measure_list(self, output, test):
            return 4

        def extract_list_metrics(self, data):
            return None

        def summarize(self, values):
            return {"mean": sum(values) / len(values)}

    acc = MeasurementCollector()
    acc.add_metric(NoneExtractMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    df = acc.list_metrics()
    assert pd.isna(df.loc["u1", "none_extract"])


# metricWrapper properties and summarization


def test_metricwrapper_is_decomposed_property():
    class DummyDecomposed(DecomposedMetric):
        label = "dummy_decomp"

        def compute_list_data(self, recs, test):
            return {"a": 1.0}

        def global_aggregate(self, values):
            return {"mean": 1.0}

        def measure_list(self, recs, test):
            return {"a": 1.0}

        def summarize(self, values):
            return {"mean": 1.0}

    wrapper = MetricWrapper(DummyDecomposed(), "decomp")
    assert wrapper.is_decomposed
    wrapper_non = MetricWrapper(ListLength(), "len")
    assert not wrapper_non.is_decomposed


def test_measure_metric_with_none_summarize():
    """Test metric that returns None from summarize."""

    class NoSummarizeMetric(Metric):
        label = "no_summarize"

        def measure_list(self, recs, test):
            return 7.0

        def summarize(self, values):
            return None

    acc = MeasurementCollector()
    acc.add_metric(NoSummarizeMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")

    summary = acc.summary_metrics()
    assert "no_summarize" not in summary


# test with movielens data


def test_full_workflow_integration_improved(ml_ds):
    split = split_temporal_fraction(ml_ds, 0.2, filter_test_users=True)
    scorer = PopScorer()
    scorer.train(split.train)
    acc = MeasurementCollector()
    acc.add_metric(Recall(10))
    acc.add_metric(NDCG(10))

    all_items = split.train.items.ids()
    test_users = list(split.test)[:5]

    for user, truth_il in test_users:
        scores = scorer(ItemList(all_items, user=[user.user_id] * len(all_items)))
        recs_il = ItemList(scores.top_n(10), user=[user.user_id] * 10, ordered=True)
        acc.measure_list(recs_il, truth_il, user=user.user_id)

    list_metrics = acc.list_metrics()
    summary = acc.summary_metrics()

    assert len(list_metrics) == len(test_users)
    assert set(list_metrics.columns) == {"Recall@10", "NDCG@10"}
    assert all(0 <= list_metrics[col].max() <= 1 for col in list_metrics.columns)

    assert len(summary) > 0
    for key, value in summary.items():
        if value is not None:
            if "std" in key.lower():
                assert value >= 0
            else:
                assert 0 <= value <= 1


# test that global metric raises errors for unsupported operations


def test_global_metric_unsupported():
    class AnotherGlobalMetric(GlobalMetric):
        label = "global"

        def measure_run(self, run, test):
            return 1.0

    metric = AnotherGlobalMetric()

    with raises(NotImplementedError, match="Global metrics don't support per-list measurement"):
        metric.measure_list(ItemList([1, 2]), ItemList([1]))

    with raises(NotImplementedError, match="Global metrics should implement measure_run instead"):
        metric.summarize([1, 2, 3])


# test edge cases in Metric.summarize


def test_list_metric_summarize_edge_cases():
    class TestListMetric(ListMetric):
        def measure_list(self, output, test):
            return len(output)

    metric = TestListMetric()

    # with empty list
    result = metric.summarize([])
    assert result["mean"] is None
    assert result["median"] is None
    assert result["std"] is None

    # with PyArrow array input
    arr = pa.array([1.0, 2.0, 3.0])
    result = metric.summarize(arr)
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["std"] == 1.0


def test_decomposed_metric_numeric_return():
    class TestDecomposedMetric(DecomposedMetric):
        def compute_list_data(self, output, test):
            return len(output)

        def global_aggregate(self, values):
            return 5.0

    metric = TestDecomposedMetric()
    result = metric.summarize([1, 2, 3])
    assert result == {"value": 5.0}


def test_empty_intermediate_values():
    class TestMetric(Metric):
        label = "test"

        def measure_list(self, recs, test):
            return None  # no intermediate data

        def summarize(self, values):
            return {"mean": 0.0}

    acc = MeasurementCollector()
    acc.add_metric(TestMetric())

    summary = acc.summary_metrics()
    assert not summary or "test" not in summary


def test_accumulator_duplicate_labels():
    acc = MeasurementCollector()
    acc.add_metric(ListLength(), label="dup")
    acc.add_metric(ListLength(), label="dup")

    with raises(RuntimeError, match="duplicate metric"):
        acc._validate_setup()


def test_accumulator_with_default_value_on_none_result():
    class NoneMetric(Metric):
        label = "none_metric"

        def measure_list(self, recs, test):
            return None

        def summarize(self, values):
            return {"mean": 123}

    acc = MeasurementCollector()
    acc.add_metric(NoneMetric(), default=99.0)
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    metrics = acc.list_metrics()
    # none should fall back to default value
    assert metrics.loc["u1", "none_metric"] == 99.0
