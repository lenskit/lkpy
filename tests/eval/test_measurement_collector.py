# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
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
from lenskit.metrics._base import FunctionMetric, GlobalMetric, ListMetric, Metric
from lenskit.metrics._collect import MeasurementCollector, _wrap_metric
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
    assert acc.list_metrics().empty
    assert not acc.summary_metrics()

    # measuring with no metrics
    acc.measure_list(ItemList([1]), ItemList([1]), user="u1")
    assert acc.list_metrics().empty
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
    assert approx(summary["N.mean"]) == 2.5


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
    wrapper = _wrap_metric(ListLength, label="ListLength")
    assert isinstance(wrapper.metric, ListLength)
    assert wrapper.label == "ListLength"


def test_wrap_metric_function_label():
    def custom_metric(recs, test):
        return len(recs)

    wrapper = _wrap_metric(custom_metric, None)
    assert wrapper.label == "custom_metric"


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
    assert acc.key_fields == expected_names
    metrics = acc.list_metrics()
    assert set(metrics.index.names) == set(expected_names)


# custom metrics and types


@mark.parametrize(
    "metric_input,expected_type",
    [
        (ListLength(), ListLength),
        (ListLength, ListLength),
        (lambda r, t: len(r), FunctionMetric),
    ],
)
def test_add_metric_types(metric_input, expected_type):
    acc = MeasurementCollector()
    acc.add_metric(metric_input)
    assert isinstance(acc._metrics[0].metric, expected_type)


def test_custom_labels():
    acc = MeasurementCollector()
    acc.add_metric(ListLength(), label="CustomLength")
    assert acc._metrics[0].label == "CustomLength"


# global, callable, scalar metrics


@mark.skip("global metrics disabled")
def test_global_metric():
    class DummyGlobalMetric(GlobalMetric):
        label = "dummy_global"

        def measure_run(self, run, test):
            return 42.0

    acc = MeasurementCollector()
    acc.add_metric(DummyGlobalMetric())
    result = acc._metrics[0].measure_run(ItemListCollection([]), ItemListCollection([]))
    assert result == 42.0
    # assert acc._metrics[0].is_global
    # assert not acc._metrics[0].is_listwise


def test_callable_metric():
    def _callable(recs, test):
        return len(recs)

    acc = MeasurementCollector()
    acc.add_metric(_callable, label="callable_metric")
    recs = ItemList([1, 2, 3])
    test_il = ItemList([1])
    result = acc._metrics[0].metric.measure_list(recs, test_il)
    assert result == 3


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
            if key.endswith(".std") or key.endswith(".n"):
                assert value >= 0
            else:
                assert 0 <= value <= 1


def test_empty_intermediate_values():
    class TestMetric(ListMetric):
        label = "test"

        def measure_list(self, recs, test):
            return None  # no intermediate data

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


def test_some_lists_none():
    class TestMetric(ListMetric):
        label = "test"

        def measure_list(self, recs, test):
            if len(recs):
                return len(recs)
            else:
                return None

    acc = MeasurementCollector()
    acc.add_metric(TestMetric())

    acc.measure_list(ItemList([3]), ItemList(), x=1)
    acc.measure_list(ItemList([]), ItemList(), x=2)
    acc.measure_list(ItemList([5, 20, 3]), ItemList(), x=3)

    lms = acc.list_metrics()
    assert lms.loc[1, "test"] == 1
    assert pd.isna(lms.loc[2, "test"])
    assert lms.loc[3, "test"] == 3

    summary = acc.summary_metrics()
    assert summary["test.mean"] == 2
