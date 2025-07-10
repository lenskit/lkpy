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
from lenskit.metrics import NDCG, MetricAccumulator, Recall
from lenskit.metrics._accum import to_metric_dict
from lenskit.metrics._base import GlobalMetric
from lenskit.metrics.basic import ListLength
from lenskit.splitting import split_temporal_fraction

_log = logging.getLogger(__name__)


def test_accumulator_initial_state():
    acc = MetricAccumulator()
    assert acc._metrics == []
    assert acc._labels == []
    assert acc._defaults == {}
    assert acc._results == []
    assert acc._intermediates == []


def test_add_metric_duplicate_label_raises():
    acc = MetricAccumulator()
    acc.add_metric(ListLength(), "length")
    with raises(ValueError):
        acc.add_metric(ListLength(), "length")


def test_basic_metric_flow():
    acc = MetricAccumulator()
    acc.add_metric(ListLength())

    acc.measure_list(ItemList([1, 2, 3], user=["u1"] * 3), ItemList([2, 3]), user="u1")
    acc.measure_list(ItemList([4, 5], user=["u2"] * 2), ItemList([5]), user="u2")

    result = acc.to_result()
    list_metrics = result.list_metrics()
    summary = result.global_metrics()

    assert isinstance(list_metrics, pd.DataFrame)
    assert len(list_metrics) == 2
    assert "N" in list_metrics.columns
    assert set(list_metrics["N"]) == {2.0, 3.0}

    assert isinstance(summary, pd.Series)
    assert "N" in summary
    assert summary["N"] == 2.5


def test_metric_dict():
    # test None input returns empty dict
    assert to_metric_dict("metric", None) == {}

    # test simple float returns dict with label as key
    assert to_metric_dict("metric", 1.23) == {"metric": 1.23}
    assert to_metric_dict("metric", 2) == {"metric": 2.0}

    # test nested dict returns flattened dict
    nested = {"a": 0.5, "b": 0.25}
    expected = {"metric.a": 0.5, "metric.b": 0.25}
    assert to_metric_dict("metric", nested) == expected

    # test unsupported type raises TypeError
    with raises(TypeError):
        to_metric_dict("metric", ["a", 0.5])


def test_unmeasured_metrics():
    # test for registered metrics but no measurements
    acc = MetricAccumulator()
    acc.add_metric(Recall(5))
    acc.add_metric(NDCG(5))

    result = acc.to_result()

    assert result.list_metrics().empty
    assert result.global_metrics()["Recall@5"] == 0.0
    assert result.global_metrics()["NDCG@5"] == 0.0


def test_metadata_keys_recorded():
    # test to determine if extra keys are recorded
    acc = MetricAccumulator()
    acc.add_metric(Recall(1))

    recs = ItemList([1], user=["u1"], ordered=True)
    truth = ItemList([1], user=["u1"])

    acc.measure_list(recs, truth, user="u1", session=1)

    df = acc.list_metrics()
    assert "user" in df.columns
    assert "session" in df.columns
    assert df.loc[0, "user"] == "u1"
    assert df.loc[0, "session"] == 1


def test_metric_accum(ml_ds):
    # test on ml ds, using metrics: recall and ndcg
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

    result = acc.to_result()
    list_metrics = result.list_metrics()
    summary = result.global_metrics()

    # per-list metrics
    assert isinstance(list_metrics, pd.DataFrame)
    assert len(list_metrics) == len(split.test)
    for metric in ["Recall@10", "NDCG@10"]:
        assert metric in list_metrics.columns
        assert np.all(list_metrics[metric] >= 0)
        assert np.all(list_metrics[metric] <= 1)

    # summary metrics
    assert isinstance(summary, pd.Series)
    for metric in ["Recall@10", "NDCG@10"]:
        assert isinstance(summary[metric], float)
        assert 0 <= summary[metric] <= 1
