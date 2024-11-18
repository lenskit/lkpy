# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from lenskit import util
from lenskit.algorithms import Algorithm, Recommender
from lenskit.data import ID, ItemList
from lenskit.parallel import invoke_progress, invoker
from lenskit.pipeline import Pipeline

from ._runner import BatchPipelineRunner

_logger = logging.getLogger(__name__)


def recommend(
    pipeline: Pipeline,
    users: Sequence[ID],
    n: int | None = None,
    candidates=None,
    *,
    n_jobs: int | None = None,
    **kwargs,
) -> dict[ID, ItemList]:
    """
    Convenience function to batch-generate recommendations from a pipeline.
    """
    if isinstance(pipeline, Algorithm):
        return legacy_recommend(pipeline, users, n, candidates, n_jobs=n_jobs, **kwargs)  # type: ignore

    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.recommend(n=n)
    outs = runner.run(pipeline, users)
    return {k.user_id: il for (k, il) in outs.output("recommendations")}  # type: ignore


def _recommend_user(algo, req):
    user, n, candidates = req

    _logger.debug("generating recommendations for %s", user)
    watch = util.Stopwatch()
    res = algo.recommend(user, n, candidates)
    _logger.debug("%s recommended %d/%s items for %s in %s", str(algo), len(res), n, user, watch)

    res["user"] = user
    res["rank"] = np.arange(1, len(res) + 1)

    return res.reset_index(drop=True)


def __standard_cand_fun(candidates):
    """
    Convert candidates from the forms accepted by :py:fun:`recommend` into
    a standard form, a function that takes a user and returns a candidate
    list.
    """
    if isinstance(candidates, dict):
        return candidates.get
    elif candidates is None:
        return lambda u: None
    else:
        return candidates


def legacy_recommend(algo, users, n, candidates=None, *, n_jobs=None, **kwargs):
    """
    Batch-recommend for multiple users.  The provided algorithm should be a
    :py:class:`algorithms.Recommender`.

    Args:
        algo: the algorithm
        users(array-like): the users to recommend for
        n(int): the number of recommendations to generate (None for unlimited)
        candidates:
            the users' candidate sets. This can be a function, in which case it will
            be passed each user ID; it can also be a dictionary, in which case user
            IDs will be looked up in it.  Pass ``None`` to use the recommender's
            built-in candidate selector (usually recommended).
        n_jobs(int):
            The number of processes to use for parallel recommendations.  Passed to
            :func:`lenskit.util.parallel.invoker`.

    Returns:
        A frame with at least the columns ``user``, ``rank``, and ``item``; possibly also
        ``score``, and any other columns returned by the recommender.
    """

    if n_jobs is None and "nprocs" in kwargs:
        n_jobs = kwargs["nprocs"]
        warnings.warn("nprocs is deprecated, use n_jobs", DeprecationWarning)

    rec_algo = Recommender.adapt(algo)
    if candidates is None and rec_algo is not algo:
        warnings.warn("no candidates provided and algo is not a recommender, unlikely to work")
    algo = rec_algo
    del rec_algo

    if "ratings" in kwargs:
        warnings.warn("Providing ratings to recommend is not supported", DeprecationWarning)

    candidates = __standard_cand_fun(candidates)

    _logger.info("recommending with %s for %d users (n_jobs=%s)", str(algo), len(users), n_jobs)
    with (
        invoke_progress(_logger, "recommending", len(users), unit="user") as progress,
        invoker(algo, _recommend_user, n_jobs=n_jobs, progress=progress) as worker,
    ):
        del algo
        timer = util.Stopwatch()
        results = worker.map((user, n, candidates(user)) for user in users)
        results = pd.concat(results, ignore_index=True, copy=False)
        timer.stop()
        time = timer.elapsed()
        rate = time / len(users)
        _logger.info(
            "recommended for %d users in %s (%.1fms per user)", len(users), timer, rate * 1000
        )

    return results
