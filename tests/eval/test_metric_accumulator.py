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
from lenskit.metrics._accum import MetricAccumulator, MetricWrapper
from lenskit.metrics._base import DecomposedMetric, GlobalMetric, ListMetric, Metric
from lenskit.metrics.basic import ListLength
from lenskit.splitting import split_temporal_fraction

_log = logging.getLogger(__name__)


@fixture
def basic_accumulator():
    acc = MetricAccumulator()
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


# initialization and defaults


def test_accumulator_initial_state():
    acc = MetricAccumulator()
    assert acc.metrics == []
    assert acc.list_metrics().empty
    assert acc.summary_metrics().empty


def test_accumulator_unmeasured_defaults(basic_accumulator):
    metric_labels = [m.label for m in basic_accumulator.metrics]
    assert "Recall@5" in metric_labels
    assert "NDCG@5" in metric_labels

    summary = basic_accumulator.summary_metrics()
    assert summary.loc["Recall@5", "mean"] == 0.0
    assert summary.loc["NDCG@5", "mean"] == 0.0


# measuring lists


def test_accumulator_measures_list_and_summary(sample_lists):
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(sample_lists["recs1"], sample_lists["test1"], user="u1")
    acc.measure_list(sample_lists["recs2"], sample_lists["test2"], user="u2")

    list_metrics = acc.list_metrics()
    summary = acc.summary_metrics()

    assert len(list_metrics) == 2
    assert set(list_metrics["N"]) == {2.0, 3.0}
    assert approx(summary.loc["N", "mean"]) == 2.5


def test_accumulator_empty_itemlists():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([]), ItemList([1, 2]), user="u1")
    acc.measure_list(ItemList([1, 2]), ItemList([]), user="u2")
    acc.measure_list(ItemList([]), ItemList([]), user="u3")

    metrics = acc.list_metrics()
    assert len(metrics) == 3
    assert all(metrics["N"] >= 0)


def test_list_metrics_no_key_fields():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())
    acc.measure_list(ItemList([1, 2]), ItemList([1]))
    metrics = acc.list_metrics()
    assert len(metrics) == 1
    assert metrics.index.names == [None]


# metric wrapping


def test_wrap_metric_with_class():
    acc = MetricAccumulator()
    wrapper = acc._wrap_metric(ListLength, label="ListLength", default=None)
    assert isinstance(wrapper.metric, ListLength)
    assert wrapper.label == "ListLength"


def test_wrap_metric_function_label():
    def custom_metric(recs, test):
        return len(recs)

    acc = MetricAccumulator()
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
    acc = MetricAccumulator()
    acc.add_metric(ListLength())
    acc.measure_list(ItemList([1, 2]), ItemList([2]), **keys)
    assert acc._key_fields == expected_names
    metrics = acc.list_metrics()
    assert set(metrics.index.names) == set(expected_names)


def test_accumulator_duplicate_labels():
    acc = MetricAccumulator()
    acc.add_metric(ListLength, label="dup")
    acc.add_metric(ListLength, label="dup")
    with raises(RuntimeError, match="duplicate metric"):
        acc._validate_setup()


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
    acc = MetricAccumulator()
    acc.add_metric(metric_input)
    assert isinstance(acc.metrics[0].metric, expected_type)


def test_custom_labels_and_defaults():
    acc = MetricAccumulator()
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

    acc = MetricAccumulator()
    acc.add_metric(DummyGlobalMetric())
    result = acc.metrics[0].measure_run(ItemListCollection([]), ItemListCollection([]))
    assert result == 42.0
    assert acc.metrics[0].is_global
    assert not acc.metrics[0].is_listwise


def test_callable_metric():
    def _callable(recs, test):
        return len(recs)

    acc = MetricAccumulator()
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
        label = "scalar_summarize"

        def measure_list(self, recs, test):
            return 5

        def summarize(self, values):
            return 99.0

    acc = MetricAccumulator()
    acc.add_metric(ScalarMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    summary = acc.summary_metrics()
    assert summary.loc["scalar_summarize", "mean"] == 99.0
    assert pd.isna(summary.loc["scalar_summarize", "median"])
    assert pd.isna(summary.loc["scalar_summarize", "std"])


# metrics returning None or mixed


def test_measure_metric_with_and_without_default():
    class NoMeasureMetric(Metric):
        label = "no_measure"

        def measure_list(self, recs, test):
            return 7.0

        def summarize(self, values):
            return None

    acc = MetricAccumulator()
    acc.add_metric(NoMeasureMetric())
    summary = acc.summary_metrics()
    assert summary.loc["no_measure", "mean"] == 0.0

    acc_default = MetricAccumulator()
    acc_default.add_metric(NoMeasureMetric(), label="no_measure_default")
    acc_default.measure_list(ItemList([1]), ItemList([1]), user="u1")
    list_metrics = acc_default.list_metrics()
    assert list_metrics.loc["u1", "no_measure_default"] == 7.0
    summary_df_default = acc_default.summary_metrics()
    assert summary_df_default.loc["no_measure_default", "mean"] == 7.0


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

    acc = MetricAccumulator()
    acc.add_metric(MixedMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    acc.measure_list(ItemList([2]), ItemList([2]), user="u2")

    df = acc.list_metrics()
    summary = acc.summary_metrics()

    assert "Mixed" in summary.index
    assert summary.loc["Mixed", "mean"] == 1.5
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

    acc = MetricAccumulator()
    acc.add_metric(NoneExtractMetric())
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    df = acc.list_metrics()
    assert df.loc["u1", "none_extract"] == 4


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


def test_wrapper_default_summarize_various_inputs():
    test_cases = [
        ([], {"mean": None, "median": None, "std": None}),
        ([None, None], {"mean": None, "median": None, "std": None}),
        ([42], {"mean": 42.0, "median": 42.0, "std": 0.0}),
        ([1, 2, 3, 4], {"mean": 2.5, "median": 2.5, "std": approx(1.291, abs=0.01)}),
        ([1, None, 2, 3], {"mean": 2.0, "median": 2.0, "std": 1.0}),
        (pa.array([1, 2, 3]), {"mean": 2.0, "median": 2.0, "std": 1.0}),
    ]
    wrapper = MetricWrapper(ListLength(), "test")
    for values, expected in test_cases:
        result = wrapper._default_summarize(values)
        for key in ["mean", "median", "std"]:
            if expected[key] is None:
                assert result[key] is None
            else:
                assert result[key] == expected[key]


def test_wrapper_default_summarize_chunked_array():
    wrapper = MetricWrapper(ListLength(), "test")
    chunked = pa.chunked_array([[1, 2], [3, 4]])
    result = wrapper._default_summarize(chunked)
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["std"] == approx(1.291, abs=0.01)


# test with movielens data


def test_full_workflow_integration_improved(ml_ds):
    split = split_temporal_fraction(ml_ds, 0.2, filter_test_users=True)
    scorer = PopScorer()
    scorer.train(split.train)
    acc = MetricAccumulator()
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
    assert set(summary.index) == {"Recall@10", "NDCG@10"}
    assert all(0 <= summary.loc[metric, "mean"] <= 1 for metric in summary.index)
    for metric in summary.index:
        assert summary.loc[metric, "mean"] is not None
        std_val = summary.loc[metric, "std"]
        assert std_val is None or std_val >= 0


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
