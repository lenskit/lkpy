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
from lenskit.metrics._base import DecomposedMetric, GlobalMetric, Metric
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


# metric accumulator tests
def test_accumulator_initial_state():
    acc = MetricAccumulator()
    assert acc.metrics == []
    assert acc.list_metrics().empty
    assert acc.summary_metrics().empty


def test_accumulator_unmeasured_defaults(basic_accumulator):
    assert len(basic_accumulator.metrics) == 2
    metric_labels = [m.label for m in basic_accumulator.metrics]
    assert "Recall@5" in metric_labels
    assert "NDCG@5" in metric_labels

    summary = basic_accumulator.summary_metrics()
    assert summary.loc["Recall@5", "mean"] == 0.0
    assert summary.loc["NDCG@5", "mean"] == 0.0


def test_accumulator_basic_flow(sample_lists):
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(sample_lists["recs1"], sample_lists["test1"], user="u1")
    acc.measure_list(sample_lists["recs2"], sample_lists["test2"], user="u2")

    list_metrics = acc.list_metrics()
    summary = acc.summary_metrics()

    assert len(list_metrics) == 2
    assert set(list_metrics["N"]) == {2.0, 3.0}
    assert approx(summary.loc["N", "mean"]) == 2.5


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


# metric types tests
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


def test_mixed_result_types():
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
    assert "Mixed" in df.columns or "Mixed.b" in df.columns
    assert len(df) == 2

    # verify the mixed data handling
    summary = acc.summary_metrics()
    assert "Mixed" in summary.index
    assert summary.loc["Mixed", "mean"] == 1.5


def test_global_and_callable_fixed():
    # global metric test
    class DummyGlobalMetric(GlobalMetric):
        def measure_run(self, run, test):
            return 123.0

    wrapper_global = MetricWrapper(DummyGlobalMetric(), "global")
    result = wrapper_global.measure_run(ItemListCollection([]), ItemListCollection([]))
    assert result == 123.0
    assert wrapper_global.is_global
    assert not wrapper_global.is_listwise

    with raises(TypeError):
        MetricWrapper(ListLength(), "N").measure_run(ItemListCollection([]), ItemListCollection([]))

    # callable metric test
    def callable_metric_func(recs, test):
        return len(recs)

    wrapper_callable = MetricWrapper(callable_metric_func, "callable")
    result_callable = wrapper_callable.measure_list(ItemList([1, 2, 3]), ItemList([1]))
    assert result_callable == 3
    assert wrapper_callable.is_listwise
    assert not wrapper_callable.is_global


def test_summarize_scalar_converts_to_dict():
    class ScalarMetric(Metric):
        label = "scalar_summarize"

        def measure_list(self, recs, test):
            return 5

        def summarize(self, values):
            return 99.0

    acc = MetricAccumulator()
    acc.add_metric(ScalarMetric(), "scalar_summarize")
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    summary = acc.summary_metrics()

    assert summary.loc["scalar_summarize", "mean"] == 99.0
    assert pd.isna(summary.loc["scalar_summarize", "median"])
    assert pd.isna(summary.loc["scalar_summarize", "std"])


def test_no_measure_metric():
    class NoMeasureMetric(Metric):
        label = "no_measure"

        def measure_list(self, recs, test):
            return None

        def summarize(self, values):
            return None

    acc = MetricAccumulator()
    acc.add_metric(NoMeasureMetric(), "no_measure")
    summary_df = acc.summary_metrics()
    assert summary_df.loc["no_measure", "mean"] == 0.0


# default summarize test
@mark.parametrize(
    "values,expected",
    [
        ([], {"mean": None, "median": None, "std": None}),
        ([None, None], {"mean": None, "median": None, "std": None}),
        ([42], {"mean": 42.0, "median": 42.0, "std": 0.0}),
        ([1, 2, 3, 4], {"mean": 2.5, "median": 2.5, "std": approx(1.291, abs=0.01)}),
        ([1, None, 2, 3], {"mean": 2.0, "median": 2.0, "std": 1.0}),
        (pa.array([1, 2, 3]), {"mean": 2.0, "median": 2.0, "std": 1.0}),
    ],
)
def test_default_summarize_various_inputs(values, expected):
    wrapper = MetricWrapper(ListLength(), "test")
    result = wrapper._default_summarize(values)

    for key in ["mean", "median", "std"]:
        if expected[key] is None:
            assert result[key] is None
        else:
            assert result[key] == expected[key]


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


def test_accumulator_empty_itemlists():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    # empty recommendations
    acc.measure_list(ItemList([]), ItemList([1, 2]), user="u1")
    # empty test set
    acc.measure_list(ItemList([1, 2]), ItemList([]), user="u2")
    # both empty
    acc.measure_list(ItemList([]), ItemList([]), user="u3")

    metrics = acc.list_metrics()
    assert len(metrics) == 3
    assert all(metrics["N"] >= 0)
