# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import logging
import warnings

import pandas as pd

from .. import util
from ..parallel import invoke_progress, invoker

_logger = logging.getLogger(__name__)


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


def predict(algo, pairs, *, n_jobs=None, **kwargs):
    """
    Generate predictions for user-item pairs.  The provided algorithm should be a
    :py:class:`algorithms.Predictor` or a function of two arguments: the user ID and
    a list of item IDs. It should return a dictionary or a :py:class:`pandas.Series`
    mapping item IDs to predictions.

    To use this function, provide a pre-fit algorithm:

        >>> from lenskit.algorithms.bias import Bias
        >>> from lenskit.metrics.predict import RMSE
        >>> from lenskit.data import from_interactions_df
        >>> from lenskit.data.movielens import load_movielens_df
        >>> ratings = load_movielens_df('data/ml-latest-small')
        >>> bias = Bias()
        >>> bias.fit(from_interactions_df(ratings[:-1000]))
        <lenskit.algorithms.bias.Bias object at ...>
        >>> preds = predict(bias, ratings[-1000:])
        >>> preds.head()
               user  item  rating   timestamp  prediction
        99004   664  8361     3.0  1393891425    3.288286
        99005   664  8528     3.5  1393891047    3.559119
        99006   664  8529     4.0  1393891173    3.573008
        99007   664  8636     4.0  1393891175    3.846268
        99008   664  8641     4.5  1393890852    3.710635
        >>> RMSE(preds)
        0.832699...

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
