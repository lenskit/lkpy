# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import numpy as np

from pytest import approx, fixture, mark, raises

from lenskit.metrics._base import Metric
from lenskit.testing import demo_recs


class DummyMetric(Metric):
    def measure_list(self, output, test, /):
        return 1.0

    def summarize(self, values, /):
        if hasattr(values, "to_pylist"):
            values = values.to_pylist()  # type: ignore

        numeric_values = [
            float(v) for v in values if isinstance(v, (int, float, np.integer, np.floating))
        ]

        if not numeric_values:
            return {"mean": None, "median": None, "std": None}

        arr = np.array(numeric_values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }


class DummyGlobalMetric(Metric):
    @property
    def label(self):
        return "DummyGlobalMetric"

    @property
    def is_global(self):
        return True

    def measure_list(self, output, test, /):
        raise NotImplementedError

    def summarize(self, values, /):
        raise NotImplementedError

    def measure_run(self, outputs, test, /):
        return 0.0


@fixture
def metric():
    return DummyMetric()  # type: ignore


@mark.parametrize(
    "input_val, expected",
    [
        (42, 42.0),
        (3.14, 3.14),
        (np.int32(7), 7.0),
        (np.float64(2.5), 2.5),
        ({"precision": 0.9}, {"precision": 0.9}),
        ("not a number", None),
        ([1, 2, 3], None),
    ],
)
def test_extract_list_metrics(metric, input_val, expected):
    assert metric.extract_list_metrics(input_val) == expected


def test_summarize(metric):
    values = [1, 2, 3]
    result = metric.summarize(values)
    assert result == {
        "mean": approx(2, abs=0.05),
        "median": 2,
        "std": approx(1, abs=0.05),
    }


def test_globalmetric(demo_recs):
    split, recs = demo_recs
    from lenskit.metrics.bulk import RunAnalysis

    bms = RunAnalysis()
    bms.add_metric(DummyGlobalMetric())

    result = bms.measure(recs, split.test)
    global_results = result.global_metrics()

    assert global_results["DummyGlobalMetric"] == 0.0


def test_listmetric():
    class LM(Metric):
        def measure_list(self, output, test, /):
            return 1.0

        def summarize(self, values, /):
            return {"mean": 1.0}

    m = LM()
    assert m.extract_list_metrics(5) == 5.0
    assert m.extract_list_metrics("non-numeric") is None


def test_decomposedmetric():
    class DM(Metric):
        def compute_list_data(self, output, test, /):
            return {"val": 4}

        def extract_list_metric(self, data, /):
            return 99

        def measure_list(self, output, test, /):
            return self.compute_list_data(output, test)

        def extract_list_metrics(self, data, /):
            return self.extract_list_metric(data)

        def summarize(self, values, /):
            result = {"agg": float(len(values))}
            return result

    dm = DM()

    result = dm.measure_list([], [])
    assert result == {"val": 4}

    extracted = dm.extract_list_metrics(result)
    assert extracted == 99

    summary = dm.summarize([{"val": 1}, {"val": 2}])
    assert summary == {"agg": 2.0}


def test_globalmetric_notimplemented():
    gm = DummyGlobalMetric()
    with raises(NotImplementedError):
        gm.measure_list([], [])
    with raises(NotImplementedError):
        gm.summarize([1, 2, 3])
