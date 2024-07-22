# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from typing_extensions import override

from lenskit.data import Dataset

from . import Predictor

_logger = logging.getLogger(__name__)


class Bias(Predictor):
    """
    A user-item bias rating prediction algorithm.  This implements the following
    predictor algorithm:

    .. math::
       s(u,i) = \\mu + b_i + b_u

    where :math:`\\mu` is the global mean rating, :math:`b_i` is item bias, and
    :math:`b_u` is the user bias.  With the provided damping values
    :math:`\\beta_{\\mathrm{u}}` and :math:`\\beta_{\\mathrm{i}}`, they are computed
    as follows:

    .. math::
       \\begin{align*}
       \\mu & = \\frac{\\sum_{r_{ui} \\in R} r_{ui}}{|R|} &
       b_i & = \\frac{\\sum_{r_{ui} \\in R_i} (r_{ui} - \\mu)}{|R_i| + \\beta_{\\mathrm{i}}} &
       b_u & = \\frac{\\sum_{r_{ui} \\in R_u} (r_{ui} - \\mu - b_i)}{|R_u| + \\beta_{\\mathrm{u}}}
       \\end{align*}

    The damping values can be interpreted as the number of default (mean) ratings to assume
    *a priori* for each user or item, damping low-information users and items towards a mean instead
    of permitting them to take on extreme values based on few ratings.

    Args:
        items: whether to compute item biases
        users: whether to compute user biases
        damping:
            Bayesian damping to apply to computed biases.  Either a number, to
            damp both user and item biases the same amount, or a (user,item) tuple
            providing separate damping values.
    """

    mean_: float
    "The global mean rating."
    item_offsets_: pd.Series | None
    "The item offsets (:math:`b_i` values)."
    user_offsets_: pd.Series | None
    "The user offsets (:math:`b_u` values)."

    def __init__(
        self, items: bool = True, users: bool = True, damping: float | tuple[float, float] = 0.0
    ):
        self.items = items
        self.users = users
        if isinstance(damping, tuple):
            self.damping = damping
            self.user_damping, self.item_damping = damping
        else:
            self.damping = damping
            self.user_damping = damping
            self.item_damping = damping

        if self.user_damping < 0:
            raise ValueError("user damping must be non-negative")
        if self.item_damping < 0:
            raise ValueError("item damping must be non-negative")

    @override
    def fit(self, data: Dataset, **kwargs):
        """
        Train the bias model on some rating data.

        Args:
            ratings (DataFrame): a data frame of ratings. Must have at least `user`,
                                 `item`, and `rating` columns.

        Returns:
            Bias: the fit bias object.
        """
        _logger.info("building bias model for %d ratings", data.interaction_count)
        ratings = data.interaction_matrix("scipy", layout="coo", field="rating")
        nrows, ncols = ratings.shape

        self.mean_ = float(np.mean(ratings.data))
        _logger.info("global mean: %.3f", self.mean_)

        centered = ratings.data - self.mean_
        if np.allclose(centered, 0):
            _logger.warn("mean-centered ratings are all 0, bias probably meaningless")

        if self.items:
            counts = np.full(ncols, self.item_damping)
            sums = np.zeros(ncols)
            np.add.at(counts, ratings.col, 1)
            np.add.at(sums, ratings.col, centered)
            means = sums / counts
            self.item_offsets_ = pd.Series(means, index=data.items.index, name="i_off")
            centered -= means[ratings.col]
            _logger.info("computed means for %d items", len(self.item_offsets_))
        else:
            self.item_offsets_ = None

        if self.users:
            counts = np.full(nrows, self.user_damping)
            sums = np.zeros(nrows)
            np.add.at(counts, ratings.row, 1)
            np.add.at(sums, ratings.row, centered)
            means = sums / counts
            self.user_offsets_ = pd.Series(means, index=data.users.index, name="u_off")
            _logger.info("computed means for %d users", len(self.user_offsets_))
        else:
            self.user_offsets_ = None

        return self

    def transform(self, ratings: pd.DataFrame, *, indexes: bool = False):
        """
        Transform ratings by removing the bias term.  This method does *not*
        recompute user (or item) biases based on these ratings, but rather uses
        the biases that were estimated with :meth:`fit`.

        Args:
            ratings:
                The ratings to transform.  Must contain at least ``user``,
                ``item``, and ``rating`` columns.
            indexes:
                if ``True``, the resulting frame will include ``uidx`` and
                ``iidx`` columns containing the 0-based user and item indexes
                for each rating.

        Returns:
            A data frame with ``rating`` transformed by subtracting user-item
            bias prediction.
        """
        rvps = ratings[["user", "item"]].copy()
        rvps["rating"] = ratings["rating"] - self.mean_
        if self.item_offsets_ is not None:
            rvps = rvps.join(self.item_offsets_, on="item", how="left")
            rvps["rating"] -= rvps["i_off"].fillna(0)
            rvps = rvps.drop(columns="i_off")
        if self.user_offsets_ is not None:
            rvps = rvps.join(self.user_offsets_, on="user", how="left")
            rvps["rating"] -= rvps["u_off"].fillna(0)
            rvps = rvps.drop(columns="u_off")
        if indexes:
            rvps["uidx"] = self.user_offsets_.index.get_indexer(rvps["user"])
            rvps["iidx"] = self.item_offsets_.index.get_indexer(rvps["item"])
        return rvps

    def inverse_transform(self, ratings):
        """
        Transform ratings by removing the bias term.
        """
        rvps = pd.DataFrame({"user": ratings["user"], "item": ratings["item"]})
        rvps["rating"] = ratings["rating"] + self.mean_
        if self.item_offsets_ is not None:
            rvps = rvps.join(self.item_offsets_, on="item", how="left")
            rvps["rating"] += rvps["i_off"].fillna(0)
            del rvps["i_off"]
        if self.user_offsets_ is not None:
            rvps = rvps.join(self.user_offsets_, on="user", how="left")
            rvps["rating"] += rvps["u_off"].fillna(0)
            del rvps["u_off"]
        return rvps

    def transform_user(self, ratings):
        """
        Transform a user's ratings by subtracting the bias model.

        Args:
            ratings(pandas.Series): The user's ratings, indexed by item.
                Must have at least `item` as index and `rating` column.

        Returns:
            pandas.Series:
                The transformed ratings and the user bias.
        """
        ratings = ratings.subtract(self.mean_)

        if self.item_offsets_ is not None:
            ioff = self.item_offsets_.reindex(ratings.index, fill_value=0)
            ratings = ratings - ioff

        u_offset = self._mean(ratings, self.user_damping)

        ratings = ratings.subtract(u_offset)
        return ratings, u_offset

    def inverse_transform_user(self, user, ratings, user_bias=None):
        """
        Un-transform a user's ratings by adding in the bias model.

        Args:
            user: The user ID.
            ratings(pandas.Series): The user's ratings, indexed by item.
            user_bias(float or None): If `None`, it looks up the user bias learned by `fit`.

        Returns:
            pandas.Series: The user's de-normalized ratings.
        """
        ratings = ratings.add(self.mean_)

        if self.item_offsets_ is not None:
            ioff = self.item_offsets_.reindex(ratings.index, fill_value=0)
            ratings = ratings + ioff

        if user_bias is not None:
            ratings = ratings.add(user_bias)
        elif self.user_offsets_ is not None and user in self.user_offsets_.index:
            ratings = ratings + self.user_offsets_.loc[user]

        return ratings

    def fit_transform(self, data: Dataset, **kwargs) -> pd.DataFrame:
        """
        Fit with ratings and return the training data transformed.
        """
        # FIXME: make this more efficient, don't rename things.
        self.fit(data)
        return self.transform(
            data.interaction_matrix("pandas", field="rating", original_ids=True).rename(
                columns={"user_id": "user", "item_id": "item"}
            ),
            **kwargs,
        )

    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.  Unknown users and items
        are assumed to have zero bias.

        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): the user's ratings (indexed by item id); if
                                 provided, will be used to recompute the user's
                                 bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        idx = pd.Index(items)
        preds = pd.Series(self.mean_, idx)

        if self.item_offsets_ is not None:
            preds = preds + self.item_offsets_.reindex(idx, fill_value=0)

        if self.users and ratings is not None:
            uoff = ratings - self.mean_
            if self.item_offsets_ is not None:
                uoff = uoff - self.item_offsets_
            umean = uoff.mean()
            preds = preds + umean
        elif self.user_offsets_ is not None:
            umean = self.user_offsets_.get(user, 0.0)
            _logger.debug("using mean(user %s) = %.3f", user, umean)
            preds = preds + umean

        return preds

    @property
    def user_index(self):
        "Get the user index from this (fit) bias."
        return self.user_offsets_.index

    @property
    def item_index(self):
        "Get the item index from this (fit) bias."
        return self.item_offsets_.index

    def _mean(self, series, damping):
        if damping is not None and damping > 0:
            return series.sum() / (series.count() + damping)
        else:
            return series.mean()

    def __str__(self):
        return "Bias(ud={}, id={})".format(self.user_damping, self.item_damping)
