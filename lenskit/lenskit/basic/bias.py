# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT
"""
Bias scoring model.
"""

from __future__ import annotations

import logging

import numpy as np

from lenskit.data import Dataset
from lenskit.data.items import ItemList
from lenskit.data.query import QueryInput, RecQuery
from lenskit.data.types import UITuple
from lenskit.data.vocab import Vocabulary
from lenskit.pipeline import Component

_logger = logging.getLogger(__name__)


class BiasScorer(Component):
    """
    A user-item bias rating prediction model.  This implements the following
    predictor function:

    .. math::
       s(u,i) = \\mu + b_i + b_u

    where :math:`\\mu` is the global mean rating, :math:`b_i` is item bias, and
    :math:`b_u` is the user bias.  With the provided damping values
    :math:`\\beta_{\\mathrm{u}}` and :math:`\\beta_{\\mathrm{i}}`, they are
    computed as follows:

    .. math::
       \\begin{align*}
       \\mu & = \\frac{\\sum_{r_{ui} \\in R} r_{ui}}{|R|} &
       b_i & = \\frac{\\sum_{r_{ui} \\in R_i} (r_{ui} - \\mu)}{|R_i| + \\beta_{\\mathrm{i}}} &
       b_u & = \\frac{\\sum_{r_{ui} \\in R_u} (r_{ui} - \\mu - b_i)}{|R_u| + \\beta_{\\mathrm{u}}}
       \\end{align*}

    The damping values can be interpreted as the number of default (mean)
    ratings to assume *a priori* for each user or item, damping low-information
    users and items towards a mean instead of permitting them to take on extreme
    values based on few ratings.

    Args:
        items:
            whether to compute item biases
        users:
            whether to compute user biases
        damping:
            Bayesian damping to apply to computed biases.  Either a number, to
            damp both user and item biases the same amount, or a (user,item)
            tuple providing separate damping values.
    """

    users: bool
    items: bool
    damping: UITuple[float]
    "The configured offset damping levels."

    mean_: float
    "The global mean rating."
    items_: Vocabulary | None = None
    item_biases_: np.ndarray[int, np.dtype[np.float32]] | None
    "The item offsets (:math:`b_i` values)."
    users_: Vocabulary | None = None
    user_biases_: np.ndarray[int, np.dtype[np.float32]] | None
    "The user offsets (:math:`b_u` values)."

    def __init__(
        self,
        items: bool = True,
        users: bool = True,
        damping: float | UITuple[float] | tuple[float, float] = 0.0,
    ):
        self.items = items
        self.users = users
        self.damping = UITuple.create(damping)

        if self.damping.user < 0:
            raise ValueError("user damping must be non-negative")
        if self.damping.item < 0:
            raise ValueError("item damping must be non-negative")

    def train(self, data: Dataset):
        """
        Train the bias model on some rating data.

        Args:
            ratings:
                The training data (must have ratings).

        Returns:
            The trained bias object.
        """
        _logger.info("building bias model for %d ratings", data.interaction_count)
        ratings = data.interaction_matrix("scipy", layout="coo", field="rating")
        nrows, ncols = ratings.shape  # type: ignore

        self.mean_ = float(np.mean(ratings.data))
        _logger.info("global mean: %.3f", self.mean_)

        centered = ratings.data - self.mean_
        if np.allclose(centered, 0):
            _logger.warning("mean-centered ratings are all 0, bias probably meaningless")

        if self.items:
            counts = np.full(ncols, self.damping.item)
            sums = np.zeros(ncols)
            np.add.at(counts, ratings.col, 1)
            np.add.at(sums, ratings.col, centered)

            # store 0 offsets
            i_bias = np.zeros(ncols, dtype=np.float32)
            np.divide(sums, counts, out=i_bias, where=counts > 0)

            self.items_ = data.items.copy()
            self.item_biases_ = i_bias
            centered -= i_bias[ratings.col]
            _logger.info("computed biases for %d items", len(self.item_biases_))
        else:
            self.item_biases_ = None

        if self.users:
            counts = np.full(nrows, self.damping.user)
            sums = np.zeros(nrows)
            np.add.at(counts, ratings.row, 1)
            np.add.at(sums, ratings.row, centered)

            u_bias = np.zeros(nrows, dtype=np.float32)
            np.divide(sums, counts, out=u_bias, where=counts > 0)

            self.users_ = data.users.copy()
            self.user_biases_ = u_bias
            _logger.info("computed biases for %d users", len(self.user_biases_))
        else:
            self.user_biases_ = None

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        """
        Compute predictions for a user and items.  Unknown users and items are
        assumed to have zero bias.

        Args:
            query:
                The recommendation query.  If the query has an item list with
                ratings, those ratings are used to compute a new bias instead of
                using the user's recorded bias.
            items:
                The items to score.
        Returns:
            Scores for `items`.
        """
        query = RecQuery.create(query)
        preds = np.full(len(items), self.mean_)

        if self.item_biases_ is not None:
            assert self.items_ is not None
            idxes = items.numbers(vocabulary=self.items_, missing="negative")
            mask = idxes >= 0
            preds[mask] += self.item_biases_[idxes[mask]]

        ratings = query.user_items.field("rating") if query.user_items is not None else None

        if self.users and ratings is not None:
            assert query.user_items is not None  # only way we can be here

            uoff = ratings - self.mean_
            if self.item_biases_ is not None:
                idxes = query.user_items.numbers(vocabulary=self.items_, missing="negative")
                found = idxes >= 0
                uoff[found] -= self.item_biases_[idxes[found]]

            umean = uoff.mean()
            preds = preds + umean

        elif query.user_id is not None and self.user_biases_ is not None:
            assert self.users_ is not None
            uno = self.users_.number(query.user_id, missing="none")
            if uno is not None:
                umean = self.user_biases_[uno]
                _logger.debug("using mean(user %s) = %.3f", query.user_id, umean)
                preds += umean

        return ItemList(items, scores=preds)

    def _mean(self, series, damping):
        if damping is not None and damping > 0:
            return series.sum() / (series.count() + damping)
        else:
            return series.mean()

    def __str__(self):
        return "Bias(ud={}, id={})".format(self.damping.user, self.damping.item)
