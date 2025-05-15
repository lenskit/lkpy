# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from lenskit.metrics.bulk import RunAnalysis
from lenskit.metrics.ranking._gini import ExposureGini, ListGini
from lenskit.testing import DemoRecs, demo_recs, pop_recs


def test_list_gini(pop_recs: DemoRecs):
    rla = RunAnalysis()
    rla.add_metric(ListGini(items=pop_recs.split.train))

    result = rla.measure(pop_recs.recommendations, pop_recs.split.test)
    rm = result.global_metrics()
    assert "ListGini" in rm.index
    # most-popular is very high Gini
    assert rm["ListGini"] < 1
    assert rm["ListGini"] > 0.90


def test_exposure_gini(pop_recs: DemoRecs):
    rla = RunAnalysis()
    rla.add_metric(ExposureGini(items=pop_recs.split.train))

    result = rla.measure(pop_recs.recommendations, pop_recs.split.test)
    rm = result.global_metrics()
    assert "ExposureGini" in rm.index
    # most-popular is very high Gini
    assert rm["ExposureGini"] < 1
    assert rm["ExposureGini"] > 0.95


def test_gini_changes(pop_recs: DemoRecs, demo_recs: DemoRecs):
    "test that randomization improves gini"

    rla = RunAnalysis()
    rla.add_metric(ListGini(items=pop_recs.split.train))
    rla.add_metric(ExposureGini(items=pop_recs.split.train))

    pop_result = rla.measure(pop_recs.recommendations, pop_recs.split.test)
    samp_result = rla.measure(demo_recs.recommendations, demo_recs.split.test)

    pop_rm = pop_result.global_metrics()
    samp_rm = samp_result.global_metrics()

    # most-popular should higher gini on both versions
    assert pop_rm["ExposureGini"] > samp_rm["ExposureGini"]
    assert pop_rm["ListGini"] > samp_rm["ListGini"]
