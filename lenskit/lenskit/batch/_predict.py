# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import warnings
from typing import Mapping

import pandas as pd

from lenskit import util
from lenskit.data import ID, GenericKey, ItemList, ItemListCollection
from lenskit.parallel import invoke_progress, invoker
from lenskit.pipeline import Pipeline

from ._runner import BatchPipelineRunner

_logger = logging.getLogger(__name__)


def predict(
    pipeline: Pipeline,
    test: ItemListCollection[GenericKey] | Mapping[ID, ItemList],
    *,
    n_jobs: int | None = None,
    **kwargs,
) -> ItemListCollection[GenericKey]:
    """
    Convenience function to batch-generate rating predictions (or other per-item
    scores) from a pipeline.
    """
    if isinstance(pipeline, Algorithm):
        return legacy_predict(pipeline, test, n_jobs=n_jobs, **kwargs)  # type: ignore

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.predict()
    outs = runner.run(pipeline, test)
    return outs.output("predictions")  # type: ignore


def _predict_user(model, req):
    user, udf = req
    watch = util.Stopwatch()
    res = model.predict_for_user(user, udf["item"])
    res = pd.DataFrame({"user": user, "item": res.index, "prediction": res.values})
    _logger.debug(
        "%s produced %d/%d predictions for %s in %s",
        model,
        res.prediction.notna().sum(),
        len(udf),
        user,
        watch,
    )
    return res


def legacy_predict(algo, pairs, *, n_jobs=None, **kwargs):
    """
    Generate predictions for user-item pairs.  The provided algorithm should be a
    :py:class:`algorithms.Predictor` or a function of two arguments: the user ID and
    a list of item IDs. It should return a dictionary or a :py:class:`pandas.Series`
    mapping item IDs to predictions.

    Args:
        algo(lenskit.algorithms.Predictor):
            A rating predictor function or algorithm.
        pairs(pandas.DataFrame):
            A data frame of (``user``, ``item``) pairs to predict for. If this frame also
            contains a ``rating`` column, it will be included in the result.
        n_jobs(int):
            The number of processes to use for parallel batch prediction.  Passed to
            :func:`lenskit.util.parallel.invoker`.

    Returns:
        pandas.DataFrame:
            a frame with columns ``user``, ``item``, and ``prediction`` containing
            the prediction results. If ``pairs`` contains a `rating` column, this
            result will also contain a `rating` column.
    """
    if n_jobs is None and "nprocs" in kwargs:
        n_jobs = kwargs["nprocs"]
        warnings.warn("nprocs is deprecated, use n_jobs", DeprecationWarning)

    nusers = pairs["user"].nunique()

    timer = util.Stopwatch()
    nusers = pairs["user"].nunique()
    with (
        invoke_progress(_logger, "predictions", nusers, unit="user") as progress,
        invoker(algo, _predict_user, n_jobs=n_jobs, progress=progress) as worker,
    ):
        del algo  # maybe free some memory

        _logger.info(
            "generating %d predictions for %d users (setup took %s)", len(pairs), nusers, timer
        )
        timer = util.Stopwatch()
        results = worker.map((user, udf.copy()) for (user, udf) in pairs.groupby("user"))
        results = pd.concat(results)
        _logger.info("generated %d predictions for %d users in %s", len(pairs), nusers, timer)

    if "rating" in pairs:
        return pairs.join(results.set_index(["user", "item"]), on=("user", "item"))
    return results
